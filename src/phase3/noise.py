from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .qiskit_qrc import assemble_feature_tensor, counts_to_features
from .qrc_parameters import ReservoirParameters, TemporalIsingQRCConfig


def build_aer_noise_model(
    depolarizing_1q: float = 0.0,
    depolarizing_2q: float = 0.0,
    amplitude_damping: float = 0.0,
):
    try:
        from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error
    except ImportError as exc:
        raise RuntimeError("Install qiskit-aer to run noise studies") from exc
    model = NoiseModel()
    if depolarizing_1q > 0:
        e1 = depolarizing_error(depolarizing_1q, 1)
        model.add_all_qubit_quantum_error(e1, ["rx", "ry", "rz"])
    if amplitude_damping > 0:
        ad = amplitude_damping_error(amplitude_damping)
        model.add_all_qubit_quantum_error(ad, ["rx", "ry", "rz"])
    if depolarizing_2q > 0:
        e2 = depolarizing_error(depolarizing_2q, 2)
        model.add_all_qubit_quantum_error(e2, ["cx", "cz", "rzz"])
    return model


def run_aer_counts(
    circuits: Sequence[object],
    mapping: Sequence[Tuple[int, int]],
    cfg: TemporalIsingQRCConfig,
    params: ReservoirParameters,
    shots: int,
    seed: int,
    noise_model=None,
    max_parallel_threads: int = 0,
) -> np.ndarray:
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise RuntimeError("Install qiskit-aer to run finite-shot/noise studies") from exc
    backend = AerSimulator(noise_model=noise_model)
    transpiled = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=seed)
    job = backend.run(
        transpiled,
        shots=int(shots),
        seed_simulator=int(seed),
        max_parallel_threads=max_parallel_threads or None,
    )
    result = job.result()
    vectors: List[np.ndarray] = []
    for idx in range(len(transpiled)):
        vectors.append(counts_to_features(result.get_counts(idx), cfg, params))
    return assemble_feature_tensor(vectors, mapping)
