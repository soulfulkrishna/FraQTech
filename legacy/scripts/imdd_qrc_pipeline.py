import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import TensorDataset

from IMDD import LCDDataset, SSMFDataset
from gaussian_qrc import GaussianQRC, GaussianQRCConfig
from readouts import LinearReadout
from tcn import TCNReadout


@dataclass
class IMDDConfig:
    dataset_name: str = "LCD"
    bit_wise: bool = False
    continuous_sampling: bool = False
    n_taps: int | None = None
    noise_power_db: float = -20.0

    total_samples: int = 10000
    train_size: int = 7000
    val_size: int = 1500
    test_size: int = 1500

    batch_size: int = 256
    num_workers: int = 0

    modes: int = 16
    virtual_nodes: int = 8
    washout: int = 0

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    seed: int = 7

    cache_path: str = "outputs/cached_features/imdd_lcd_cache.pt"

@torch.no_grad()
def materialize_split(dataset, device):
    """
    Consume an IMDD split exactly once and turn it into tensors.
    Returns:
        X: [N, T, 1]
        Y: [N]
    """
    xs, ys = [], []
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    for x, y in loader:
        xs.append(x)
        ys.append(y.long())

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    return X, Y


def normalize_tensor_split(X, mean, std):
    return (X - mean) / (std + 1e-8)


@torch.no_grad()
def cache_tensor_split(qrc, X, Y, batch_size, device):
    qrc.eval()
    zs, ys = [], []

    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=False)

    for x, y in loader:
        x = x.to(device)
        z = qrc(x)
        zs.append(z.cpu())
        ys.append(y.cpu())

    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)

class IMDDWrappedDataset(Dataset):
    """
    Wrap IMDD samples so input becomes [T, 1] and label is a scalar class index.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]

        # x is expected to be [T]; make it [T, 1]
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        if x.dim() != 1:
            raise ValueError(f"Expected IMDD input chunk shape [T], got {tuple(x.shape)}")

        x = x.unsqueeze(-1)  # [T, 1]

        # y is expected to be scalar class label
        if torch.is_tensor(y):
            y = y.long().reshape(())
        else:
            y = torch.tensor(y, dtype=torch.long)

        return x, y

def make_imdd_dataset(cfg: IMDDConfig):
    name = cfg.dataset_name.upper()

    if name == "LCD":
        ds = LCDDataset()
        default_n_taps = 7
    elif name == "SSMF":
        ds = SSMFDataset()
        default_n_taps = 21
    else:
        raise ValueError(f"Unsupported dataset_name={cfg.dataset_name}")

    n_taps = cfg.n_taps if cfg.n_taps is not None else default_n_taps
    ds.set_n_taps(n_taps)
    ds.set_noise_power_db(cfg.noise_power_db)

    if hasattr(ds, "continuous_sampling"):
        ds.continuous_sampling = cfg.continuous_sampling
    if hasattr(ds, "bit_level"):
        ds.bit_level = cfg.bit_wise
    if hasattr(ds, "bit_wise"):
        ds.bit_wise = cfg.bit_wise

    wrapped = IMDDWrappedDataset(ds)

    if cfg.total_samples > len(wrapped):
        raise ValueError(
            f"Requested total_samples={cfg.total_samples}, but dataset length is only {len(wrapped)}"
        )

    subset, _ = random_split(
        wrapped,
        [cfg.total_samples, len(wrapped) - cfg.total_samples],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_ds, val_ds, test_ds = random_split(
        subset,
        [cfg.train_size, cfg.val_size, cfg.test_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    return train_ds, val_ds, test_ds


@torch.no_grad()
def compute_input_stats(loader, device):
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for x, _ in loader:
        x = x.to(device)  # [B, T, 1]
        total_sum += x.sum().item()
        total_sq_sum += (x * x).sum().item()
        total_count += x.numel()

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean * mean
    std = max(var, 1e-8) ** 0.5
    return mean, std


class NormalizedDataset(Dataset):
    def __init__(self, dataset, mean: float, std: float):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = (x - self.mean) / (self.std + 1e-8)
        return x, y


@torch.no_grad()
def cache_split(qrc, loader, device):
    qrc.eval()
    all_z = []
    all_y = []

    for x, y in loader:
        x = x.to(device)  # [B, T, 1]
        z = qrc(x)        # [B, T', F]

        all_z.append(z.cpu())
        all_y.append(y.long().cpu())

    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0)


class CachedSeqToOneDataset(Dataset):
    def __init__(self, z, y):
        self.z = z
        self.y = y

    def __len__(self):
        return self.z.size(0)

    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]


class LastStepLinearReadout(nn.Module):
    """
    Sequence-to-one classifier from cached reservoir features.
    Uses only the last time step of z.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, F]
        z_last = z[:, -1, :]
        return self.proj(z_last)  # [B, C]


class LastStepTCNReadout(nn.Module):
    """
    Sequence-to-one classifier using a causal TCN over cached z.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        channels: int = 32,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tcn = TCNReadout(
            in_dim=in_dim,
            out_dim=out_dim,
            channels=channels,
            levels=levels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits_seq = self.tcn(z)   # [B, T, C]
        return logits_seq[:, -1, :]  # [B, C]

class PureTCNClassifier(nn.Module):
    """
    Sequence-to-one classifier on raw IM/DD input x: [B, T, 1].
    Uses the same causal TCN backbone and last-step pooling style as QRC+TCN.
    """
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 4,
        channels: int = 32,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tcn = TCNReadout(
            in_dim=in_dim,
            out_dim=out_dim,
            channels=channels,
            levels=levels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_seq = self.tcn(x)     # [B, T, C]
        return logits_seq[:, -1, :]  # [B, C]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for z, y in loader:
        z = z.to(device)
        y = y.to(device)

        logits = model(z)         # [B, C]
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * z.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_count += y.numel()

    return total_loss / len(loader.dataset), total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for z, y in loader:
        z = z.to(device)
        y = y.to(device)

        logits = model(z)
        loss = criterion(logits, y)

        total_loss += loss.item() * z.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_count += y.numel()

    return total_loss / len(loader.dataset), total_correct / total_count

def train_one_epoch_seq(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)   # [B, C]
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_count += y.numel()

    return total_loss / len(loader.dataset), total_correct / total_count


@torch.no_grad()
def evaluate_seq(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_count += y.numel()

    return total_loss / len(loader.dataset), total_correct / total_count

def infer_num_classes(labels: torch.Tensor) -> int:
    return int(labels.max().item()) + 1

def summarize_results(rows, label):
        vals = torch.tensor([r[label] for r in rows], dtype=torch.float32)
        return vals.mean().item(), vals.std(unbiased=False).item()

def main():
        seeds = [7, 11, 19, 23, 31]
        settings = [
            {"dataset_name": "LCD", "n_taps": None, "noise_power_db": -20.0},
            {"dataset_name": "SSMF", "n_taps": None, "noise_power_db": -20.0},
        ]

        for setting in settings:
            print("\n" + "=" * 80)
            print(f"RUNNING {setting['dataset_name']} @ noise {setting['noise_power_db']} dB")
            print("=" * 80)

            rows = []
            for seed in seeds:
                cfg = IMDDConfig(
                    dataset_name=setting["dataset_name"],
                    n_taps=setting["n_taps"],
                    noise_power_db=setting["noise_power_db"],
                    seed=seed,
                    cache_path=f"outputs/cached_features/{setting['dataset_name'].lower()}_seed{seed}.pt",
                )
                row = run_experiment(cfg)
                rows.append(row)

            lin_mean, lin_std = summarize_results(rows, "linear_test_acc")
            qrc_mean, qrc_std = summarize_results(rows, "qrc_tcn_test_acc")
            pure_mean, pure_std = summarize_results(rows, "pure_tcn_test_acc")

            print("\n" + "-" * 80)
            print(f"{setting['dataset_name']} summary over {len(seeds)} seeds")
            print(f"QRC+Linear : {lin_mean:.4f} ± {lin_std:.4f}")
            print(f"QRC+TCN    : {qrc_mean:.4f} ± {qrc_std:.4f}")
            print(f"Pure TCN   : {pure_mean:.4f} ± {pure_std:.4f}")
            print("-" * 80)
            
def run_experiment(cfg: IMDDConfig):

    os.makedirs("outputs/cached_features", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------
    # 1) Load IMDD dataset
    # -------------------------
    train_raw, val_raw, test_raw = make_imdd_dataset(cfg)

    # -------------------------
    # 2) Materialize each split exactly once
    # -------------------------
    train_X, train_Y = materialize_split(train_raw, device)
    val_X, val_Y = materialize_split(val_raw, device)
    test_X, test_Y = materialize_split(test_raw, device)

    # -------------------------
    # 3) Normalize using train stats only
    # -------------------------
    mean = train_X.mean().item()
    std = train_X.std(unbiased=False).item()
    print(f"Train normalization: mean={mean:.6f}, std={std:.6f}")

    train_X = normalize_tensor_split(train_X, mean, std)
    val_X = normalize_tensor_split(val_X, mean, std)
    test_X = normalize_tensor_split(test_X, mean, std)
    raw_train_loader = DataLoader(
        TensorDataset(train_X, train_Y),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    raw_val_loader = DataLoader(
        TensorDataset(val_X, val_Y),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    raw_test_loader = DataLoader(
        TensorDataset(test_X, test_Y),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    # -------------------------
    # 4) Frozen QRC + caching
    # -------------------------
    qrc_cfg = GaussianQRCConfig(
        input_dim=1,
        modes=cfg.modes,
        virtual_nodes=cfg.virtual_nodes,
        washout=cfg.washout,
        include_means=True,
        include_squares=True,
        include_abs=True,
        include_pair_products=True,
        include_variances=False,
        include_covariances=False,
        gate_strength=0.35,
        cov_gate_strength=0.20,
        mean_nonlinearity="tanh",
    )
    qrc = GaussianQRC(qrc_cfg).to(device)

    train_z, train_y = cache_tensor_split(qrc, train_X, train_Y, cfg.batch_size, device)
    val_z, val_y = cache_tensor_split(qrc, val_X, val_Y, cfg.batch_size, device)
    test_z, test_y = cache_tensor_split(qrc, test_X, test_Y, cfg.batch_size, device)

    num_classes = infer_num_classes(train_y)
    feature_dim = train_z.shape[-1]

    payload = {
        "train_z": train_z,
        "train_y": train_y,
        "val_z": val_z,
        "val_y": val_y,
        "test_z": test_z,
        "test_y": test_y,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "imdd_config": vars(cfg),
        "qrc_config": vars(qrc_cfg),
        "train_mean": mean,
        "train_std": std,
    }
    torch.save(payload, cfg.cache_path)

    print(f"Saved cache to: {cfg.cache_path}")
    print("train_z:", train_z.shape, "train_y:", train_y.shape)
    print("val_z  :", val_z.shape, "val_y  :", val_y.shape)
    print("test_z :", test_z.shape, "test_y :", test_y.shape)
    print("feature_dim:", feature_dim, "num_classes:", num_classes)

    # -------------------------
    # 5) Build cached datasets/loaders
    # -------------------------
    train_cached = CachedSeqToOneDataset(train_z, train_y)
    val_cached = CachedSeqToOneDataset(val_z, val_y)
    test_cached = CachedSeqToOneDataset(test_z, test_y)

    train_loader = DataLoader(
        train_cached, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_cached, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_cached, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    criterion = nn.CrossEntropyLoss()

    print("\n=== Training QRC + Linear ===")
    linear_model = LastStepLinearReadout(feature_dim, num_classes).to(device)
    linear_opt = torch.optim.AdamW(
        linear_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_linear_val = 0.0
    best_linear_state = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(linear_model, train_loader, linear_opt, criterion, device)
        va_loss, va_acc = evaluate(linear_model, val_loader, criterion, device)

        if va_acc > best_linear_val:
            best_linear_val = va_acc
            best_linear_state = {k: v.cpu() for k, v in linear_model.state_dict().items()}

        print(
            f"[Linear] Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
            f"best_val_acc={best_linear_val:.4f}"
        )

    linear_model.load_state_dict(best_linear_state)
    te_linear_loss, te_linear_acc = evaluate(linear_model, test_loader, criterion, device)
    print(f"[Linear] TEST | loss={te_linear_loss:.4f} acc={te_linear_acc:.4f}")

    torch.save(
        best_linear_state,
        f"outputs/checkpoints/{cfg.dataset_name.lower()}_seed{cfg.seed}_qrc_linear.pt",
    )

    # -------------------------
    # 5) Train cached TCN readout
    # -------------------------
    print("\n=== Training QRC + TCN ===")
    tcn_model = LastStepTCNReadout(
        in_dim=feature_dim,
        out_dim=num_classes,
        channels=32,
        levels=4,
        kernel_size=3,
        dropout=0.1,
    ).to(device)

    tcn_opt = torch.optim.AdamW(
        tcn_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_tcn_val = 0.0
    best_tcn_state = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(tcn_model, train_loader, tcn_opt, criterion, device)
        va_loss, va_acc = evaluate(tcn_model, val_loader, criterion, device)

        if va_acc > best_tcn_val:
            best_tcn_val = va_acc
            best_tcn_state = {k: v.cpu() for k, v in tcn_model.state_dict().items()}

        print(
            f"[TCN]    Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
            f"best_val_acc={best_tcn_val:.4f}"
        )

    tcn_model.load_state_dict(best_tcn_state)
    te_qrc_tcn_loss, te_qrc_tcn_acc = evaluate(tcn_model, test_loader, criterion, device)
    print(f"[TCN] TEST | loss={te_qrc_tcn_loss:.4f} acc={te_qrc_tcn_acc:.4f}")

    torch.save(
        best_tcn_state,
        f"outputs/checkpoints/{cfg.dataset_name.lower()}_seed{cfg.seed}_qrc_tcn.pt",
    )

    # -------------------------
    # 6) Train Pure TCN on raw IM/DD input
    # -------------------------
    print("\n=== Training Pure TCN ===")
    pure_tcn_model = PureTCNClassifier(
        in_dim=1,
        out_dim=num_classes,
        channels=32,
        levels=4,
        kernel_size=3,
        dropout=0.1,
    ).to(device)

    pure_tcn_opt = torch.optim.AdamW(
        pure_tcn_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_pure_tcn_val = 0.0
    best_pure_tcn_state = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch_seq(
            pure_tcn_model, raw_train_loader, pure_tcn_opt, criterion, device
        )
        va_loss, va_acc = evaluate_seq(
            pure_tcn_model, raw_val_loader, criterion, device
        )

        if va_acc > best_pure_tcn_val:
            best_pure_tcn_val = va_acc
            best_pure_tcn_state = {
                k: v.cpu() for k, v in pure_tcn_model.state_dict().items()
            }

        print(
            f"[PureTCN] Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
            f"best_val_acc={best_pure_tcn_val:.4f}"
        )

    pure_tcn_model.load_state_dict(best_pure_tcn_state)
    te_pure_tcn_loss, te_pure_tcn_acc = evaluate_seq(
        pure_tcn_model, raw_test_loader, criterion, device
    )
    print(f"[PureTCN] TEST | loss={te_pure_tcn_loss:.4f} acc={te_pure_tcn_acc:.4f}")
    torch.save(
        best_pure_tcn_state,
        f"outputs/checkpoints/{cfg.dataset_name.lower()}_seed{cfg.seed}_pure_tcn.pt",
    )
    return {
        "dataset": cfg.dataset_name,
        "seed": cfg.seed,
        "linear_test_acc": te_linear_acc,
        "qrc_tcn_test_acc": te_qrc_tcn_acc,
        "pure_tcn_test_acc": te_pure_tcn_acc,
    }

    
if __name__ == "__main__":
    main()