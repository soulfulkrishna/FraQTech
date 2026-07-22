import math
from typing import Dict

import torch
import torch.nn as nn

from src.models_qml.quantum_layers import PennyLaneQuantumLayer


class QNNRegressor(nn.Module):
    """
    Window-level hybrid QNN regressor.

    Raw window -> classical compression -> variational quantum layer -> linear head.
    """

    def __init__(
        self,
        lookback: int,
        input_dim: int,
        n_qubits: int,
        n_layers: int,
        circuit_type: str,
        hidden_dim: int,
        shots: int = 0,
    ) -> None:
        super().__init__()

        self.lookback = lookback
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        self.hidden_dim = hidden_dim
        self.shots = shots

        self.pre = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh(),
        )

        self.q_layer = PennyLaneQuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
            shots=shots,
        )

        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.pre(x) * math.pi
        q_features = self.q_layer(angles)
        return self.head(q_features).squeeze(-1)

    def trainable_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def total_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def circuit_evals_per_sample(self) -> int:
        return 1

    def circuit_depth_proxy(self) -> int:
        return self.q_layer.circuit_depth_proxy()


class QRNNRegressor(nn.Module):
    """
    Hybrid QRNN-style recurrent model.

    At each time step:
        [x_t, h_{t-1}] -> quantum layer -> GRUCell -> h_t
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int,
        n_layers: int,
        circuit_type: str,
        hidden_dim: int,
        shots: int = 0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        self.hidden_dim = hidden_dim
        self.shots = shots

        self.pre = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, n_qubits),
            nn.Tanh(),
        )

        self.q_layer = PennyLaneQuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
            shots=shots,
        )

        self.cell = nn.GRUCell(input_size=n_qubits, hidden_size=hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        h = torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            xt = x[:, t, :]
            q_in = self.pre(torch.cat([xt, h], dim=-1)) * math.pi
            q_features = self.q_layer(q_in)
            h = self.cell(q_features, h)

        return self.head(h).squeeze(-1)

    def trainable_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def total_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def circuit_evals_per_sample(self, lookback: int) -> int:
        return int(lookback)

    def circuit_depth_proxy(self) -> int:
        return self.q_layer.circuit_depth_proxy()


class QLSTMRegressor(nn.Module):
    """
    Hybrid QLSTM-style recurrent model.

    At each time step:
        [x_t, h_{t-1}] -> quantum layer -> LSTMCell -> h_t, c_t
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int,
        n_layers: int,
        circuit_type: str,
        hidden_dim: int,
        shots: int = 0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        self.hidden_dim = hidden_dim
        self.shots = shots

        self.pre = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, n_qubits),
            nn.Tanh(),
        )

        self.q_layer = PennyLaneQuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
            shots=shots,
        )

        self.cell = nn.LSTMCell(input_size=n_qubits, hidden_size=hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        h = torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)
        c = torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            xt = x[:, t, :]
            q_in = self.pre(torch.cat([xt, h], dim=-1)) * math.pi
            q_features = self.q_layer(q_in)
            h, c = self.cell(q_features, (h, c))

        return self.head(h).squeeze(-1)

    def trainable_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def total_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def circuit_evals_per_sample(self, lookback: int) -> int:
        return int(lookback)

    def circuit_depth_proxy(self) -> int:
        return self.q_layer.circuit_depth_proxy()


def build_qml_model(
    config: Dict,
    lookback: int,
    input_dim: int,
) -> nn.Module:
    model_type = config["model_type"]

    if model_type == "qnn":
        return QNNRegressor(
            lookback=lookback,
            input_dim=input_dim,
            n_qubits=int(config["n_qubits"]),
            n_layers=int(config["n_layers"]),
            circuit_type=str(config["circuit_type"]),
            hidden_dim=int(config["hidden_dim"]),
            shots=int(config.get("shots", 0)),
        )

    if model_type == "qrnn":
        return QRNNRegressor(
            input_dim=input_dim,
            n_qubits=int(config["n_qubits"]),
            n_layers=int(config["n_layers"]),
            circuit_type=str(config["circuit_type"]),
            hidden_dim=int(config["hidden_dim"]),
            shots=int(config.get("shots", 0)),
        )

    if model_type == "qlstm":
        return QLSTMRegressor(
            input_dim=input_dim,
            n_qubits=int(config["n_qubits"]),
            n_layers=int(config["n_layers"]),
            circuit_type=str(config["circuit_type"]),
            hidden_dim=int(config["hidden_dim"]),
            shots=int(config.get("shots", 0)),
        )

    raise ValueError(f"Unsupported QML model_type={model_type}")