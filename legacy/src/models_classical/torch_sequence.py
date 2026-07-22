from typing import Optional, Dict

import numpy as np


class TorchSequenceRegressor:
    """
    Unified wrapper for MLP, RNN, GRU, LSTM, TCN, and Transformer regressors.

    Input shape:
        X: (N, lookback, channels)

    Output:
        y: (N,)
    """

    def __init__(
        self,
        model_type: str,
        input_dim: int = 1,
        lookback: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 20,
        seed: int = 0,
        device: Optional[str] = None,
    ) -> None:
        import torch

        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model = self._build_model()
        self.model.to(self.device)

    def _build_model(self):
        if self.model_type == "mlp":
            return MLPRegressor(
                lookback=self.lookback,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )

        if self.model_type in {"rnn", "gru", "lstm"}:
            return RecurrentRegressor(
                cell_type=self.model_type,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )

        if self.model_type == "tcn":
            return TCNRegressor(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )

        if self.model_type == "transformer":
            return TransformerRegressor(
                input_dim=self.input_dim,
                lookback=self.lookback,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )

        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TorchSequenceRegressor":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        dataset = TensorDataset(X_tensor, y_tensor)

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        self.model.train()

        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X_test, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.model.eval()

        preds = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[start : start + self.batch_size]).to(self.device)
                pred = self.model(xb).detach().cpu().numpy().reshape(-1)
                preds.append(pred)

        out = np.concatenate(preds)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out.astype(float)

    def trainable_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def total_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def config(self) -> Dict[str, object]:
        return {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "lookback": self.lookback,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "device": self.device,
        }


class MLPRegressor:
    def __new__(cls, lookback: int, input_dim: int, hidden_dim: int, dropout: float):
        import torch.nn as nn

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )


def _get_recurrent_layer(cell_type: str):
    import torch.nn as nn

    if cell_type == "rnn":
        return nn.RNN
    if cell_type == "gru":
        return nn.GRU
    if cell_type == "lstm":
        return nn.LSTM

    raise ValueError(f"Unknown recurrent cell_type: {cell_type}")


class RecurrentRegressor:
    def __new__(
        cls,
        cell_type: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        import torch.nn as nn

        recurrent_cls = _get_recurrent_layer(cell_type)

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()

                rnn_dropout = dropout if num_layers > 1 else 0.0

                self.rnn = recurrent_cls(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=rnn_dropout,
                )

                self.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x):
                out, _ = self.rnn(x)
                last = out[:, -1, :]
                return self.head(last)

        return _Model()


class CausalConvBlock:
    def __new__(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()

                self.padding = (kernel_size - 1) * dilation

                self.conv1 = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=self.padding,
                )
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout)

                self.conv2 = nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=self.padding,
                )
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout)

                if in_channels != out_channels:
                    self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
                else:
                    self.residual = nn.Identity()

            def _crop(self, x):
                if self.padding > 0:
                    return x[:, :, :-self.padding]
                return x

            def forward(self, x):
                out = self.conv1(x)
                out = self._crop(out)
                out = self.relu1(out)
                out = self.dropout1(out)

                out = self.conv2(out)
                out = self._crop(out)
                out = self.relu2(out)
                out = self.dropout2(out)

                res = self.residual(x)

                return out + res

        return _Block()


class TCNRegressor:
    def __new__(
        cls,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()

                blocks = []
                channels_in = input_dim

                for i in range(num_layers):
                    dilation = 2 ** i
                    blocks.append(
                        CausalConvBlock(
                            in_channels=channels_in,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            dilation=dilation,
                            dropout=dropout,
                        )
                    )
                    channels_in = hidden_dim

                self.network = nn.Sequential(*blocks)
                self.head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                # x: (batch, time, channels) -> (batch, channels, time)
                x = x.transpose(1, 2)
                out = self.network(x)
                last = out[:, :, -1]
                return self.head(last)

        return _Model()


class TransformerRegressor:
    def __new__(
        cls,
        input_dim: int,
        lookback: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()

                if hidden_dim % 4 != 0:
                    raise ValueError("hidden_dim must be divisible by 4 for nhead=4.")

                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self.positional = nn.Parameter(torch.zeros(1, lookback, hidden_dim))

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 2,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )

                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=num_layers,
                )

                self.head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                h = self.input_proj(x)
                h = h + self.positional[:, : h.shape[1], :]
                h = self.encoder(h)
                last = h[:, -1, :]
                return self.head(last)

        return _Model()