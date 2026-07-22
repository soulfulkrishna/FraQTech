import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import torch

from src.energy.resource_tracker import ResourceTracker
from src.models_qrc.qrc_factory import build_qrc_encoder


def stable_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_cache_paths(
    project_root: Path,
    dataset: str,
    config_name: str,
    seed: int,
    config: Dict[str, Any],
    lookback: int,
    horizon: int,
) -> Tuple[Path, Path]:
    cache_dir = project_root / "data" / "qrc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    h = stable_hash(
        {
            "dataset": dataset,
            "config_name": config_name,
            "seed": seed,
            "config": config,
            "lookback": lookback,
            "horizon": horizon,
        }
    )

    feature_path = cache_dir / f"{dataset}_{config_name}_seed{seed}_{h}.pt"
    meta_path = cache_dir / f"{dataset}_{config_name}_seed{seed}_{h}.json"

    return feature_path, meta_path


@torch.no_grad()
def encode_in_batches(
    encoder: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    encoder.eval()

    X_t = torch.as_tensor(X, dtype=torch.float32)
    chunks = []

    for start in range(0, len(X_t), batch_size):
        xb = X_t[start : start + batch_size].to(device)
        zb = encoder(xb).detach().cpu()
        chunks.append(zb)

    return torch.cat(chunks, dim=0)


def get_or_create_qrc_cache(
    project_root: Path,
    dataset: str,
    config_name: str,
    config: Dict[str, Any],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lookback: int,
    horizon: int,
    device: torch.device,
    batch_size: int,
    track_energy: bool,
    force_recache: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], ResourceTracker]:
    feature_path, meta_path = make_cache_paths(
        project_root=project_root,
        dataset=dataset,
        config_name=config_name,
        seed=seed,
        config=config,
        lookback=lookback,
        horizon=horizon,
    )

    if feature_path.exists() and meta_path.exists() and not force_recache:
        payload = torch.load(feature_path, map_location="cpu")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        dummy = ResourceTracker(track_energy=False)
        dummy.elapsed_time_sec = 0.0
        dummy.energy_kwh = 0.0
        dummy.carbon_kgco2e = 0.0
        dummy.peak_ram_gb = 0.0
        dummy.peak_gpu_mem_gb = 0.0

        meta["loaded_from_cache"] = True

        return (
            payload["Z_train"],
            payload["y_train"],
            payload["Z_test"],
            payload["y_test"],
            meta,
            dummy,
        )

    encoder, encoder_meta = build_qrc_encoder(
        cfg=config,
        seed=seed,
        device=device,
    )

    with ResourceTracker(track_energy=track_energy, project_name="qrc-cache") as tracker:
        Z_train = encode_in_batches(
            encoder=encoder,
            X=X_train,
            device=device,
            batch_size=batch_size,
        )

        Z_test = encode_in_batches(
            encoder=encoder,
            X=X_test,
            device=device,
            batch_size=batch_size,
        )

    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    y_test_t = torch.as_tensor(y_test, dtype=torch.float32)

    payload = {
        "Z_train": Z_train,
        "y_train": y_train_t,
        "Z_test": Z_test,
        "y_test": y_test_t,
    }

    torch.save(payload, feature_path)

    meta = {
        "dataset": dataset,
        "config_name": config_name,
        "seed": seed,
        "lookback": lookback,
        "horizon": horizon,
        "encoder_meta": encoder_meta,
        "feature_path": str(feature_path),
        "loaded_from_cache": False,
        "Z_train_shape": list(Z_train.shape),
        "Z_test_shape": list(Z_test.shape),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return Z_train, y_train_t, Z_test, y_test_t, meta, tracker