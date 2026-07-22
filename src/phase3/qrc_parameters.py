from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TemporalIsingQRCConfig:
    n_qubits: int = 5
    input_dim: int = 1
    temporal_bins: int = 8
    virtual_nodes: int = 2
    reservoir_layers: int = 1
    topology: str = "ring_chord"
    input_scale_y: float = 0.9
    input_scale_z: float = 0.55
    field_x_scale: float = 0.45
    field_z_scale: float = 0.25
    interaction_scale: float = 0.38
    input_reupload: bool = True
    include_z: bool = True
    include_zz: bool = True
    include_global_features: bool = True
    washout_bins: int = 0
    parameter_seed_offset: int = 31000

    def validate(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >=2")
        if self.temporal_bins < 1 or self.virtual_nodes < 1 or self.reservoir_layers < 1:
            raise ValueError("temporal_bins, virtual_nodes, and reservoir_layers must be positive")
        if not (self.include_z or self.include_zz or self.include_global_features):
            raise ValueError("Enable at least one feature family")


@dataclass(frozen=True)
class ReservoirParameters:
    input_y: np.ndarray
    input_z: np.ndarray
    bias_y: np.ndarray
    bias_z: np.ndarray
    field_x: np.ndarray
    field_z: np.ndarray
    couplings: np.ndarray
    edges: Tuple[Tuple[int, int], ...]
    seed: int

    def to_serializable(self) -> Dict[str, object]:
        return {
            "input_y": self.input_y.tolist(),
            "input_z": self.input_z.tolist(),
            "bias_y": self.bias_y.tolist(),
            "bias_z": self.bias_z.tolist(),
            "field_x": self.field_x.tolist(),
            "field_z": self.field_z.tolist(),
            "couplings": self.couplings.tolist(),
            "edges": [list(e) for e in self.edges],
            "seed": self.seed,
        }


def topology_edges(n_qubits: int, topology: str) -> Tuple[Tuple[int, int], ...]:
    ring = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    if topology == "ring":
        return tuple(ring)
    if topology == "line":
        return tuple((q, q + 1) for q in range(n_qubits - 1))
    if topology == "ring_chord":
        chords = []
        if n_qubits >= 5:
            step = max(2, n_qubits // 3)
            for q in range(n_qubits):
                j = (q + step) % n_qubits
                edge = tuple(sorted((q, j)))
                if edge[0] != edge[1] and edge not in chords and edge not in [tuple(sorted(e)) for e in ring]:
                    chords.append(edge)
        return tuple(ring + chords)
    raise ValueError(f"Unsupported topology={topology}")


def generate_reservoir_parameters(cfg: TemporalIsingQRCConfig, seed: int) -> ReservoirParameters:
    cfg.validate()
    actual_seed = int(seed + cfg.parameter_seed_offset)
    rng = np.random.default_rng(actual_seed)
    edges = topology_edges(cfg.n_qubits, cfg.topology)
    # Quasi-random signs improve diversity while keeping angles hardware-friendly.
    input_y = rng.normal(0.0, cfg.input_scale_y, size=(cfg.n_qubits, cfg.input_dim))
    input_z = rng.normal(0.0, cfg.input_scale_z, size=(cfg.n_qubits, cfg.input_dim))
    bias_y = rng.uniform(-0.20, 0.20, size=cfg.n_qubits)
    bias_z = rng.uniform(-0.20, 0.20, size=cfg.n_qubits)
    field_x = rng.normal(0.0, cfg.field_x_scale, size=(cfg.reservoir_layers, cfg.n_qubits))
    field_z = rng.normal(0.0, cfg.field_z_scale, size=(cfg.reservoir_layers, cfg.n_qubits))
    couplings = rng.normal(0.0, cfg.interaction_scale, size=(cfg.reservoir_layers, len(edges)))
    return ReservoirParameters(
        input_y=input_y.astype(np.float64),
        input_z=input_z.astype(np.float64),
        bias_y=bias_y.astype(np.float64),
        bias_z=bias_z.astype(np.float64),
        field_x=field_x.astype(np.float64),
        field_z=field_z.astype(np.float64),
        couplings=couplings.astype(np.float64),
        edges=edges,
        seed=actual_seed,
    )


def feature_dimension(cfg: TemporalIsingQRCConfig, edges: Sequence[Tuple[int, int]] | None = None) -> int:
    edges = tuple(edges) if edges is not None else topology_edges(cfg.n_qubits, cfg.topology)
    dim = 0
    if cfg.include_z:
        dim += cfg.n_qubits
    if cfg.include_zz:
        dim += len(edges)
    if cfg.include_global_features:
        dim += 4  # mean Z, std Z, mean |Z|, mean ZZ
    return dim
