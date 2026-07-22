from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .data import piecewise_aggregate_approximation
from .qrc_parameters import (
    ReservoirParameters,
    TemporalIsingQRCConfig,
    generate_reservoir_parameters,
)


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
    except ImportError as exc:
        raise RuntimeError(
            "Qiskit is required for finite-shot/QPU execution. Install requirements-qpu.txt."
        ) from exc
    return QuantumCircuit, SparsePauliOp


def pauli_label(n_qubits: int, qubits: Sequence[int]) -> str:
    chars = ["I"] * n_qubits
    for q in qubits:
        chars[n_qubits - 1 - int(q)] = "Z"  # Qiskit label is q_{n-1}...q_0
    return "".join(chars)


def qrc_observables(cfg: TemporalIsingQRCConfig, params: ReservoirParameters):
    _, SparsePauliOp = _require_qiskit()
    labels: List[str] = []
    if cfg.include_z or cfg.include_global_features:
        labels.extend(pauli_label(cfg.n_qubits, [q]) for q in range(cfg.n_qubits))
    if cfg.include_zz or cfg.include_global_features:
        labels.extend(pauli_label(cfg.n_qubits, [i, j]) for i, j in params.edges)
    return labels, [SparsePauliOp(label) for label in labels]


def _append_input_encoding(qc, u: np.ndarray, cfg: TemporalIsingQRCConfig, p: ReservoirParameters, vnode: int):
    if vnode > 0 and not cfg.input_reupload:
        return
    scale = 1.0 if vnode == 0 else 1.0 / float(cfg.virtual_nodes)
    ay = scale * (p.input_y @ u + p.bias_y)
    az = scale * (p.input_z @ u + p.bias_z)
    for q in range(cfg.n_qubits):
        qc.ry(float(ay[q]), q)
        qc.rz(float(az[q]), q)


def _append_reservoir(qc, cfg: TemporalIsingQRCConfig, p: ReservoirParameters):
    substep = 1.0 / float(cfg.virtual_nodes)
    for layer in range(cfg.reservoir_layers):
        for q in range(cfg.n_qubits):
            qc.rx(float(substep * p.field_x[layer, q]), q)
            qc.rz(float(substep * p.field_z[layer, q]), q)
        edge_indices = list(range(len(p.edges)))
        if layer % 2:
            edge_indices.reverse()
        for e in edge_indices:
            i, j = p.edges[e]
            qc.rzz(float(substep * p.couplings[layer, e]), i, j)


def build_checkpoint_circuit(
    window: np.ndarray,
    checkpoint: int,
    cfg: TemporalIsingQRCConfig,
    params: ReservoirParameters,
    measure: bool = False,
):
    QuantumCircuit, _ = _require_qiskit()
    window = np.asarray(window, dtype=float)
    if window.shape != (cfg.temporal_bins, cfg.input_dim):
        raise ValueError(
            f"Window must have shape {(cfg.temporal_bins, cfg.input_dim)}, got {window.shape}"
        )
    max_checkpoint = cfg.temporal_bins * cfg.virtual_nodes - 1
    if checkpoint < 0 or checkpoint > max_checkpoint:
        raise ValueError(f"checkpoint must be in [0,{max_checkpoint}]")
    qc = QuantumCircuit(cfg.n_qubits)
    current = -1
    for t in range(cfg.temporal_bins):
        for vnode in range(cfg.virtual_nodes):
            _append_input_encoding(qc, window[t], cfg, params, vnode)
            _append_reservoir(qc, cfg, params)
            current += 1
            if current == checkpoint:
                if measure:
                    qc.measure_all()
                return qc
    raise RuntimeError("Checkpoint construction failed")


def build_checkpoint_circuits(
    X: np.ndarray,
    cfg: TemporalIsingQRCConfig,
    seed: int,
    measure: bool = False,
) -> Tuple[List[object], List[Tuple[int, int]], ReservoirParameters]:
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] != cfg.temporal_bins:
        X = piecewise_aggregate_approximation(X, cfg.temporal_bins)
    params = generate_reservoir_parameters(cfg, seed)
    circuits: List[object] = []
    mapping: List[Tuple[int, int]] = []
    start_checkpoint = cfg.washout_bins * cfg.virtual_nodes
    total = cfg.temporal_bins * cfg.virtual_nodes
    for sample_idx, window in enumerate(X):
        for checkpoint in range(start_checkpoint, total):
            circuits.append(build_checkpoint_circuit(window, checkpoint, cfg, params, measure=measure))
            mapping.append((sample_idx, checkpoint - start_checkpoint))
    return circuits, mapping, params


def expectation_vector_to_features(
    evs: np.ndarray,
    cfg: TemporalIsingQRCConfig,
    params: ReservoirParameters,
) -> np.ndarray:
    evs = np.asarray(evs, dtype=float).reshape(-1)
    cursor = 0
    z = evs[cursor : cursor + cfg.n_qubits]
    cursor += cfg.n_qubits
    zz = evs[cursor : cursor + len(params.edges)]
    chunks = []
    if cfg.include_z:
        chunks.append(z)
    if cfg.include_zz:
        chunks.append(zz)
    if cfg.include_global_features:
        chunks.append(
            np.asarray(
                [
                    float(np.mean(z)),
                    float(np.std(z)),
                    float(np.mean(np.abs(z))),
                    float(np.mean(zz)),
                ],
                dtype=float,
            )
        )
    return np.concatenate(chunks).astype(np.float32)


def counts_to_expectations(
    counts: Dict[str, int],
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    shots = float(sum(counts.values()))
    if shots <= 0:
        raise ValueError("Counts are empty")
    z = np.zeros(n_qubits, dtype=float)
    zz = np.zeros(len(edges), dtype=float)
    for raw, count in counts.items():
        bits = raw.replace(" ", "")[::-1]  # qubit 0 first
        signs = np.asarray([1.0 if bits[q] == "0" else -1.0 for q in range(n_qubits)])
        z += count * signs
        for e, (i, j) in enumerate(edges):
            zz[e] += count * signs[i] * signs[j]
    return z / shots, zz / shots


def counts_to_features(
    counts: Dict[str, int],
    cfg: TemporalIsingQRCConfig,
    params: ReservoirParameters,
) -> np.ndarray:
    z, zz = counts_to_expectations(counts, cfg.n_qubits, params.edges)
    chunks = []
    if cfg.include_z:
        chunks.append(z)
    if cfg.include_zz:
        chunks.append(zz)
    if cfg.include_global_features:
        chunks.append(np.asarray([z.mean(), z.std(), np.abs(z).mean(), zz.mean()]))
    return np.concatenate(chunks).astype(np.float32)


def assemble_feature_tensor(
    vectors: Sequence[np.ndarray],
    mapping: Sequence[Tuple[int, int]],
) -> np.ndarray:
    if len(vectors) != len(mapping):
        raise ValueError("vectors and mapping lengths differ")
    n_samples = max(i for i, _ in mapping) + 1
    n_checkpoints = max(c for _, c in mapping) + 1
    feature_dim = len(vectors[0])
    out = np.empty((n_samples, n_checkpoints, feature_dim), dtype=np.float32)
    for vector, (sample_idx, checkpoint) in zip(vectors, mapping):
        out[sample_idx, checkpoint] = vector
    return out


def circuit_resource_summary(circuits: Sequence[object]) -> Dict[str, object]:
    depths = [int(c.depth() or 0) for c in circuits]
    sizes = [int(c.size()) for c in circuits]
    twoq = []
    for c in circuits:
        counts = c.count_ops()
        twoq.append(int(sum(v for k, v in counts.items() if k in {"cx", "cz", "ecr", "rzz", "rxx", "ryy"})))
    return {
        "num_circuits": len(circuits),
        "logical_depth_min": int(min(depths)) if depths else 0,
        "logical_depth_mean": float(np.mean(depths)) if depths else 0.0,
        "logical_depth_max": int(max(depths)) if depths else 0,
        "logical_gates_mean": float(np.mean(sizes)) if sizes else 0.0,
        "logical_two_qubit_gates_mean": float(np.mean(twoq)) if twoq else 0.0,
    }
