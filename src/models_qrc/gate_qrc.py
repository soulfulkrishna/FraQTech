import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class GateQRCConfig:
    input_dim: int = 1
    n_qubits: int = 6
    virtual_nodes: int = 2
    washout: int = 2

    input_scale: float = 1.0
    reservoir_layers: int = 2
    random_angle_scale: float = 0.35
    entangle: str = "ring"

    include_z: bool = True
    include_zz: bool = True
    include_state_probs: bool = False

    shots: int = 0
    dtype: torch.dtype = torch.complex64


class GateQRC(nn.Module):
    """
    Lightweight gate-based statevector QRC simulator.

    Input:
        x: [B, T, input_dim]

    Output:
        z: [B, T - washout, feature_dim]

    This is simulated-only. It uses exact statevector expectations unless
    shots > 0 is later implemented.
    """

    def __init__(self, cfg: GateQRCConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.state_dim = 2 ** cfg.n_qubits

        gen = torch.Generator()
        gen.manual_seed(seed)

        reservoir_ry = torch.randn(
            cfg.reservoir_layers,
            cfg.n_qubits,
            generator=gen,
        ) * cfg.random_angle_scale

        reservoir_rz = torch.randn(
            cfg.reservoir_layers,
            cfg.n_qubits,
            generator=gen,
        ) * cfg.random_angle_scale

        input_weights = torch.randn(
            cfg.n_qubits,
            cfg.input_dim,
            generator=gen,
        ) * cfg.input_scale

        self.register_buffer("reservoir_ry", reservoir_ry.float())
        self.register_buffer("reservoir_rz", reservoir_rz.float())
        self.register_buffer("input_weights", input_weights.float())

        self.idx0_idx1 = self._make_qubit_index_pairs(cfg.n_qubits)
        self.cnot_perms = self._make_cnot_perms(cfg.n_qubits)
        self.z_signs = self._make_z_signs(cfg.n_qubits)

        self.zz_pairs = self._make_zz_pairs(cfg.n_qubits)
        self.feature_dim = self._compute_feature_dim()

        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _make_qubit_index_pairs(n_qubits: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pairs = []
        dim = 2 ** n_qubits

        for q in range(n_qubits):
            idx0 = []
            idx1 = []
            mask = 1 << q

            for basis in range(dim):
                if (basis & mask) == 0:
                    idx0.append(basis)
                    idx1.append(basis | mask)

            pairs.append(
                (
                    torch.tensor(idx0, dtype=torch.long),
                    torch.tensor(idx1, dtype=torch.long),
                )
            )

        return pairs

    @staticmethod
    def _make_cnot_perms(n_qubits: int) -> List[torch.Tensor]:
        dim = 2 ** n_qubits
        perms = []

        for control in range(n_qubits):
            target = (control + 1) % n_qubits
            c_mask = 1 << control
            t_mask = 1 << target

            perm = []
            for basis in range(dim):
                out = basis ^ t_mask if (basis & c_mask) else basis
                perm.append(out)

            perms.append(torch.tensor(perm, dtype=torch.long))

        return perms

    @staticmethod
    def _make_z_signs(n_qubits: int) -> torch.Tensor:
        dim = 2 ** n_qubits
        signs = torch.zeros(n_qubits, dim, dtype=torch.float32)

        for q in range(n_qubits):
            mask = 1 << q
            for basis in range(dim):
                signs[q, basis] = -1.0 if (basis & mask) else 1.0

        return signs

    @staticmethod
    def _make_zz_pairs(n_qubits: int) -> List[Tuple[int, int]]:
        pairs = []
        for q in range(n_qubits):
            pairs.append((q, (q + 1) % n_qubits))
        return pairs

    def _compute_feature_dim(self) -> int:
        d = 0

        if self.cfg.include_z:
            d += self.cfg.n_qubits

        if self.cfg.include_zz:
            d += len(self.zz_pairs)

        if self.cfg.include_state_probs:
            d += self.state_dim

        return d * self.cfg.virtual_nodes

    def _initial_state(self, batch_size: int, device) -> torch.Tensor:
        psi = torch.zeros(batch_size, self.state_dim, dtype=self.cfg.dtype, device=device)
        psi[:, 0] = 1.0 + 0.0j
        return psi

    def _apply_ry(self, psi: torch.Tensor, qubit: int, angles: torch.Tensor) -> torch.Tensor:
        idx0, idx1 = self.idx0_idx1[qubit]
        idx0 = idx0.to(psi.device)
        idx1 = idx1.to(psi.device)

        old0 = psi[:, idx0]
        old1 = psi[:, idx1]

        c = torch.cos(angles / 2.0).to(old0.dtype).unsqueeze(-1)
        s = torch.sin(angles / 2.0).to(old0.dtype).unsqueeze(-1)

        new0 = c * old0 - s * old1
        new1 = s * old0 + c * old1

        out = psi.clone()
        out[:, idx0] = new0
        out[:, idx1] = new1

        return out

    def _apply_rz(self, psi: torch.Tensor, qubit: int, angles: torch.Tensor) -> torch.Tensor:
        signs = self.z_signs[qubit].to(psi.device)
        phase = torch.exp((-0.5j * angles.unsqueeze(-1).to(torch.complex64)) * signs.unsqueeze(0))
        return psi * phase.to(psi.dtype)

    def _apply_cnot_ring(self, psi: torch.Tensor) -> torch.Tensor:
        out = psi

        for perm in self.cnot_perms:
            perm = perm.to(out.device)
            new_out = torch.empty_like(out)
            new_out[:, perm] = out
            out = new_out

        return out

    def _normalize(self, psi: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(torch.abs(psi) ** 2, dim=-1, keepdim=True)).clamp_min(1e-12)
        return psi / norm

    def _extract_features(self, psi: torch.Tensor) -> torch.Tensor:
        probs = torch.abs(psi) ** 2
        features = []

        if self.cfg.include_z:
            z_signs = self.z_signs.to(psi.device)
            z = probs @ z_signs.T
            features.append(z)

        if self.cfg.include_zz:
            z_signs = self.z_signs.to(psi.device)
            zz_vals = []
            for i, j in self.zz_pairs:
                zz_sign = z_signs[i] * z_signs[j]
                zz_vals.append(probs @ zz_sign)
            features.append(torch.stack(zz_vals, dim=-1))

        if self.cfg.include_state_probs:
            features.append(probs)

        if not features:
            raise ValueError("At least one gate-QRC feature family must be enabled.")

        z = torch.cat(features, dim=-1)
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _reservoir_step(self, psi: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # Input injection through RY rotations.
        input_angles = u_t @ self.input_weights.T.to(u_t.device)

        for q in range(self.cfg.n_qubits):
            psi = self._apply_ry(psi, q, input_angles[:, q])

        # Fixed random reservoir layers.
        for layer in range(self.cfg.reservoir_layers):
            for q in range(self.cfg.n_qubits):
                ry_angle = self.reservoir_ry[layer, q].to(u_t.device).expand(u_t.shape[0])
                rz_angle = self.reservoir_rz[layer, q].to(u_t.device).expand(u_t.shape[0])

                psi = self._apply_ry(psi, q, ry_angle)
                psi = self._apply_rz(psi, q, rz_angle)

            if self.cfg.entangle == "ring":
                psi = self._apply_cnot_ring(psi)
            elif self.cfg.entangle == "none":
                pass
            else:
                raise ValueError(f"Unsupported entangle={self.cfg.entangle}")

        return self._normalize(psi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, T, D], got {tuple(x.shape)}")

        B, T, D = x.shape

        if D != self.cfg.input_dim:
            raise ValueError(f"Expected input_dim={self.cfg.input_dim}, got {D}")

        x = x.float()
        psi = self._initial_state(batch_size=B, device=x.device)

        all_z = []

        for t in range(T):
            u_t = x[:, t, :]

            vnode_features = []

            for _ in range(self.cfg.virtual_nodes):
                psi = self._reservoir_step(psi, u_t)
                vnode_features.append(self._extract_features(psi))

            if t >= self.cfg.washout:
                all_z.append(torch.cat(vnode_features, dim=-1))

        if not all_z:
            raise ValueError(f"Sequence length {T} too short for washout={self.cfg.washout}.")

        return torch.stack(all_z, dim=1)

    def total_params(self) -> int:
        return int(
            self.reservoir_ry.numel()
            + self.reservoir_rz.numel()
            + self.input_weights.numel()
        )

    def trainable_params(self) -> int:
        return 0

    def circuit_depth_proxy(self) -> int:
        # Rough per-time-step depth proxy.
        return int(
            self.cfg.virtual_nodes
            * (
                self.cfg.n_qubits
                + self.cfg.reservoir_layers * (2 * self.cfg.n_qubits + self.cfg.n_qubits)
            )
        )