import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn


@dataclass
class GaussianQRCConfig:
    input_dim: int = 1
    modes: int = 8
    virtual_nodes: int = 4
    washout: int = 2

    input_scale: float = 0.5
    noise_scale: float = 0.02
    spectral_radius: float = 0.92
    leak_rate: float = 1.0

    include_means: bool = True
    include_variances: bool = True
    include_covariances: bool = False
    covariance_pairs: Optional[List[Tuple[int, int]]] = None

    include_squares: bool = False
    include_abs: bool = False
    include_pair_products: bool = False
    pair_stride: int = 2

    gate_strength: float = 0.35
    cov_gate_strength: float = 0.20
    mean_nonlinearity: str = "tanh"

    dtype: torch.dtype = torch.float32


class GaussianQRC(nn.Module):
    """
    Project-native CV/Gaussian QRC encoder.

    Input:
        x: [B, T, input_dim]

    Output:
        z: [B, T - washout, feature_dim]

    This is a frozen moment-based reservoir:
        m_t: Gaussian mean vector, shape [B, 2M]
        V_t: Gaussian covariance matrix, shape [B, 2M, 2M]

    It is designed for sustainable benchmarking, not as a full Hilbert-space
    simulator.
    """

    def __init__(self, cfg: GaussianQRCConfig):
        super().__init__()
        self.cfg = cfg
        self.state_dim = 2 * cfg.modes

        A = self._make_stable_A(
            dim=self.state_dim,
            spectral_radius=cfg.spectral_radius,
            dtype=cfg.dtype,
        )

        B = torch.randn(self.state_dim, cfg.input_dim, dtype=cfg.dtype) * cfg.input_scale
        c = torch.zeros(self.state_dim, dtype=cfg.dtype)

        q_diag = torch.full((self.state_dim,), cfg.noise_scale, dtype=cfg.dtype)
        Q = torch.diag(q_diag)

        Wg = torch.randn(self.state_dim, cfg.input_dim, dtype=cfg.dtype) * 0.5
        Wq = torch.randn(self.state_dim, cfg.input_dim, dtype=cfg.dtype) * 0.25

        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("c", c)
        self.register_buffer("Q", Q)
        self.register_buffer("Wg", Wg)
        self.register_buffer("Wq", Wq)
        self.register_buffer("base_q_diag", q_diag.clone())

        self.pair_feature_indices = self._default_feature_pairs(
            self.state_dim,
            step=cfg.pair_stride,
        )

        if cfg.include_covariances:
            if cfg.covariance_pairs is None:
                self.covariance_pairs = self._default_covariance_pairs(self.state_dim)
            else:
                self.covariance_pairs = cfg.covariance_pairs
        else:
            self.covariance_pairs = []

        self.feature_dim = self._compute_feature_dim()

        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _make_stable_A(dim: int, spectral_radius: float, dtype: torch.dtype) -> torch.Tensor:
        raw = torch.randn(dim, dim, dtype=dtype) / math.sqrt(dim)
        svals = torch.linalg.svdvals(raw)
        max_sv = svals.max().clamp_min(1e-6)
        return spectral_radius * raw / max_sv

    @staticmethod
    def _default_covariance_pairs(state_dim: int) -> List[Tuple[int, int]]:
        pairs = []
        for i in range(0, state_dim - 1, 2):
            pairs.append((i, i + 1))
        for i in range(0, state_dim - 2, 2):
            pairs.append((i, i + 2))
        return pairs

    @staticmethod
    def _default_feature_pairs(state_dim: int, step: int = 2) -> List[Tuple[int, int]]:
        pairs = []
        for i in range(0, state_dim - 1, step):
            pairs.append((i, i + 1))
        for i in range(0, state_dim - 2, step):
            pairs.append((i, i + 2))
        return pairs

    def _apply_mean_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.mean_nonlinearity == "tanh":
            return torch.tanh(x)
        if self.cfg.mean_nonlinearity == "sin":
            return torch.sin(x)
        if self.cfg.mean_nonlinearity == "relu":
            return torch.relu(x)
        raise ValueError(f"Unsupported mean_nonlinearity={self.cfg.mean_nonlinearity}")

    def _compute_feature_dim(self) -> int:
        d = 0

        if self.cfg.include_means:
            d += self.state_dim

        if self.cfg.include_variances:
            d += self.state_dim

        if self.cfg.include_covariances:
            d += len(self.covariance_pairs)

        if self.cfg.include_squares:
            d += self.state_dim

        if self.cfg.include_abs:
            d += self.state_dim

        if self.cfg.include_pair_products:
            d += len(self.pair_feature_indices)

        return d * self.cfg.virtual_nodes

    def _extract_features(self, m: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        parts = []

        if self.cfg.include_means:
            parts.append(m)

        if self.cfg.include_variances:
            variances = torch.diagonal(V, dim1=-2, dim2=-1)
            parts.append(variances)

        if self.cfg.include_covariances:
            covs = []
            for i, j in self.covariance_pairs:
                covs.append(V[:, i, j])
            parts.append(torch.stack(covs, dim=-1))

        if self.cfg.include_squares:
            parts.append(m ** 2)

        if self.cfg.include_abs:
            parts.append(torch.abs(m))

        if self.cfg.include_pair_products:
            prods = []
            for i, j in self.pair_feature_indices:
                prods.append(m[:, i] * m[:, j])
            parts.append(torch.stack(prods, dim=-1))

        if not parts:
            raise ValueError("At least one QRC feature family must be enabled.")

        z = torch.cat(parts, dim=-1)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, T, D], got {tuple(x.shape)}")

        B, T, D = x.shape

        if D != self.cfg.input_dim:
            raise ValueError(f"Expected input_dim={self.cfg.input_dim}, got {D}")

        device = x.device
        dtype = self.A.dtype

        x = x.to(dtype)

        m = torch.zeros(B, self.state_dim, device=device, dtype=dtype)
        V = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

        all_z = []

        A = self.A
        Bmat = self.B
        c = self.c
        Wg = self.Wg
        Wq = self.Wq
        base_q_diag = self.base_q_diag

        for t in range(T):
            u = x[:, t, :]

            vnode_features = []

            for _ in range(self.cfg.virtual_nodes):
                drive = u @ Bmat.T

                gate = 1.0 + self.cfg.gate_strength * torch.tanh(u @ Wg.T)
                pre_m = (m @ A.T) * gate + drive + c

                m_candidate = self._apply_mean_nonlinearity(pre_m)
                m = (1.0 - self.cfg.leak_rate) * m + self.cfg.leak_rate * m_candidate

                q_gate = 1.0 + self.cfg.cov_gate_strength * torch.sigmoid(u @ Wq.T)
                q_diag = base_q_diag.unsqueeze(0) * q_gate
                Q_batch = torch.diag_embed(q_diag)

                AV = torch.matmul(A.unsqueeze(0), V)
                V_candidate = torch.matmul(AV, A.T.unsqueeze(0)) + Q_batch
                V_candidate = 0.5 * (V_candidate + V_candidate.transpose(-1, -2))

                V = (1.0 - self.cfg.leak_rate) * V + self.cfg.leak_rate * V_candidate

                diag = torch.diagonal(V, dim1=-2, dim2=-1)
                diag = torch.clamp(diag, min=1e-8, max=1e6)
                V = V.clone()
                idx = torch.arange(self.state_dim, device=device)
                V[:, idx, idx] = diag

                m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
                V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

                vnode_features.append(self._extract_features(m, V))

            if t >= self.cfg.washout:
                all_z.append(torch.cat(vnode_features, dim=-1))

        if not all_z:
            raise ValueError(f"Sequence length {T} is too short for washout={self.cfg.washout}.")

        return torch.stack(all_z, dim=1)

    def total_params(self) -> int:
        total = 0
        for buffer in self.buffers():
            total += buffer.numel()
        return int(total)

    def trainable_params(self) -> int:
        return 0