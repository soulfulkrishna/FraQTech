from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data import piecewise_aggregate_approximation
from .qrc_parameters import (
    ReservoirParameters,
    TemporalIsingQRCConfig,
    feature_dimension,
    generate_reservoir_parameters,
)


class TemporalIsingQRC(nn.Module):
    """Frozen, hardware-portable temporal Ising quantum reservoir.

    The model performs repeated angle encoding followed by fixed transverse/longitudinal
    fields and ZZ interactions. Only the downstream classical readout is trained. Exact
    statevector expectations are used in this simulator; the matching Qiskit circuit builder
    in `qiskit_qrc.py` is used for finite-shot, noise, and QPU execution.
    """

    def __init__(
        self,
        cfg: TemporalIsingQRCConfig,
        seed: int = 0,
        parameters: ReservoirParameters | None = None,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.seed = int(seed)
        self.params = parameters or generate_reservoir_parameters(cfg, seed)
        self.complex_dtype = dtype
        self.state_dim = 1 << cfg.n_qubits
        self.edges = tuple(self.params.edges)
        self.feature_dim = feature_dimension(cfg, self.edges)

        self.register_buffer("input_y", torch.tensor(self.params.input_y, dtype=torch.float32))
        self.register_buffer("input_z", torch.tensor(self.params.input_z, dtype=torch.float32))
        self.register_buffer("bias_y", torch.tensor(self.params.bias_y, dtype=torch.float32))
        self.register_buffer("bias_z", torch.tensor(self.params.bias_z, dtype=torch.float32))
        self.register_buffer("field_x", torch.tensor(self.params.field_x, dtype=torch.float32))
        self.register_buffer("field_z", torch.tensor(self.params.field_z, dtype=torch.float32))
        self.register_buffer("couplings", torch.tensor(self.params.couplings, dtype=torch.float32))

        idx_pairs = self._make_index_pairs(cfg.n_qubits)
        for q, (idx0, idx1) in enumerate(idx_pairs):
            self.register_buffer(f"idx0_{q}", idx0)
            self.register_buffer(f"idx1_{q}", idx1)
        self.register_buffer("z_signs", self._make_z_signs(cfg.n_qubits))
        self.register_buffer("edge_signs", self._make_edge_signs(self.z_signs, self.edges))

    @staticmethod
    def _make_index_pairs(n_qubits: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dim = 1 << n_qubits
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for q in range(n_qubits):
            mask = 1 << q
            idx0 = [basis for basis in range(dim) if (basis & mask) == 0]
            idx1 = [basis | mask for basis in idx0]
            out.append((torch.tensor(idx0, dtype=torch.long), torch.tensor(idx1, dtype=torch.long)))
        return out

    @staticmethod
    def _make_z_signs(n_qubits: int) -> torch.Tensor:
        dim = 1 << n_qubits
        basis = torch.arange(dim, dtype=torch.long)
        signs = []
        for q in range(n_qubits):
            signs.append(torch.where((basis & (1 << q)) == 0, 1.0, -1.0))
        return torch.stack(signs, dim=0).float()

    @staticmethod
    def _make_edge_signs(z_signs: torch.Tensor, edges: Tuple[Tuple[int, int], ...]) -> torch.Tensor:
        return torch.stack([z_signs[i] * z_signs[j] for i, j in edges], dim=0).float()

    def _initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        psi = torch.zeros((batch_size, self.state_dim), dtype=self.complex_dtype, device=device)
        psi[:, 0] = 1.0 + 0.0j
        return psi

    def _pair(self, q: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return getattr(self, f"idx0_{q}"), getattr(self, f"idx1_{q}")

    def _apply_ry(self, psi: torch.Tensor, q: int, theta: torch.Tensor) -> torch.Tensor:
        idx0, idx1 = self._pair(q)
        old0, old1 = psi[:, idx0], psi[:, idx1]
        c = torch.cos(theta / 2.0).to(psi.real.dtype).unsqueeze(-1).to(psi.dtype)
        s = torch.sin(theta / 2.0).to(psi.real.dtype).unsqueeze(-1).to(psi.dtype)
        out = psi.clone()
        out[:, idx0] = c * old0 - s * old1
        out[:, idx1] = s * old0 + c * old1
        return out

    def _apply_rx(self, psi: torch.Tensor, q: int, theta: torch.Tensor) -> torch.Tensor:
        idx0, idx1 = self._pair(q)
        old0, old1 = psi[:, idx0], psi[:, idx1]
        c = torch.cos(theta / 2.0).unsqueeze(-1).to(psi.dtype)
        s = (-1.0j * torch.sin(theta / 2.0)).unsqueeze(-1).to(psi.dtype)
        out = psi.clone()
        out[:, idx0] = c * old0 + s * old1
        out[:, idx1] = s * old0 + c * old1
        return out

    def _apply_rz(self, psi: torch.Tensor, q: int, theta: torch.Tensor) -> torch.Tensor:
        signs = self.z_signs[q].to(psi.device)
        phase = torch.exp(-0.5j * theta[:, None].to(torch.complex64) * signs[None, :])
        return psi * phase.to(psi.dtype)

    def _apply_rzz(self, psi: torch.Tensor, edge_index: int, theta: torch.Tensor) -> torch.Tensor:
        signs = self.edge_signs[edge_index].to(psi.device)
        phase = torch.exp(-0.5j * theta[:, None].to(torch.complex64) * signs[None, :])
        return psi * phase.to(psi.dtype)

    def _inject(self, psi: torch.Tensor, u: torch.Tensor, vnode: int) -> torch.Tensor:
        if vnode > 0 and not self.cfg.input_reupload:
            return psi
        scale = 1.0 if vnode == 0 else 1.0 / float(self.cfg.virtual_nodes)
        ay = scale * (u @ self.input_y.T + self.bias_y[None, :])
        az = scale * (u @ self.input_z.T + self.bias_z[None, :])
        for q in range(self.cfg.n_qubits):
            psi = self._apply_ry(psi, q, ay[:, q])
            psi = self._apply_rz(psi, q, az[:, q])
        return psi

    def _evolve(self, psi: torch.Tensor) -> torch.Tensor:
        batch = psi.shape[0]
        substep = 1.0 / float(self.cfg.virtual_nodes)
        for layer in range(self.cfg.reservoir_layers):
            for q in range(self.cfg.n_qubits):
                tx = (substep * self.field_x[layer, q]).expand(batch)
                tz = (substep * self.field_z[layer, q]).expand(batch)
                psi = self._apply_rx(psi, q, tx)
                psi = self._apply_rz(psi, q, tz)
            # Alternate edge order across layers to reduce systematic ordering bias.
            indices = range(len(self.edges)) if layer % 2 == 0 else reversed(range(len(self.edges)))
            for e in indices:
                theta = (substep * self.couplings[layer, e]).expand(batch)
                psi = self._apply_rzz(psi, e, theta)
        return psi

    def _features(self, psi: torch.Tensor) -> torch.Tensor:
        probs = torch.abs(psi) ** 2
        z = probs @ self.z_signs.T.to(psi.device)
        zz = probs @ self.edge_signs.T.to(psi.device)
        chunks = []
        if self.cfg.include_z:
            chunks.append(z)
        if self.cfg.include_zz:
            chunks.append(zz)
        if self.cfg.include_global_features:
            chunks.append(
                torch.stack(
                    [
                        z.mean(dim=1),
                        z.std(dim=1, unbiased=False),
                        z.abs().mean(dim=1),
                        zz.mean(dim=1),
                    ],
                    dim=1,
                )
            )
        return torch.nan_to_num(torch.cat(chunks, dim=1), nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got {tuple(x.shape)}")
        if x.shape[2] != self.cfg.input_dim:
            raise ValueError(f"Expected input_dim={self.cfg.input_dim}, got {x.shape[2]}")
        if x.shape[1] != self.cfg.temporal_bins:
            raise ValueError(
                f"Expected temporal_bins={self.cfg.temporal_bins}; compress raw windows before encoding."
            )
        x = x.float()
        psi = self._initial_state(x.shape[0], x.device)
        outputs = []
        for t in range(self.cfg.temporal_bins):
            for vnode in range(self.cfg.virtual_nodes):
                psi = self._inject(psi, x[:, t, :], vnode)
                psi = self._evolve(psi)
                if t >= self.cfg.washout_bins:
                    outputs.append(self._features(psi))
        return torch.stack(outputs, dim=1)

    def metadata(self) -> Dict[str, object]:
        gates_per_checkpoint = 2 * self.cfg.n_qubits
        gates_per_checkpoint += self.cfg.reservoir_layers * (2 * self.cfg.n_qubits + len(self.edges))
        checkpoints = (self.cfg.temporal_bins - self.cfg.washout_bins) * self.cfg.virtual_nodes
        return {
            "config": asdict(self.cfg),
            "parameter_seed": self.params.seed,
            "edges": [list(e) for e in self.edges],
            "state_dimension": self.state_dim,
            "feature_dim_per_checkpoint": self.feature_dim,
            "checkpoints": checkpoints,
            "flattened_feature_dim": checkpoints * self.feature_dim,
            "logical_gate_count_full_window": gates_per_checkpoint * checkpoints,
            "logical_depth_proxy_full_window": (
                4 + 2 * self.cfg.reservoir_layers + len(self.edges)
            ) * checkpoints,
            "trainable_quantum_parameters": 0,
        }


def encode_exact(
    X: np.ndarray,
    cfg: TemporalIsingQRCConfig,
    seed: int,
    batch_size: int = 16,
    device: str = "auto",
) -> Tuple[np.ndarray, Dict[str, object]]:
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] != cfg.temporal_bins:
        X = piecewise_aggregate_approximation(X, cfg.temporal_bins)
    if device == "auto":
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected = torch.device(device)
    model = TemporalIsingQRC(cfg, seed=seed).to(selected)
    chunks = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(selected)
        chunks.append(model(xb).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0), model.metadata()
