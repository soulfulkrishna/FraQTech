from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .qiskit_qrc import (
    assemble_feature_tensor,
    counts_to_features,
    expectation_vector_to_features,
    qrc_observables,
)
from .qrc_parameters import ReservoirParameters, TemporalIsingQRCConfig


def _backend_name(backend) -> str:
    value = getattr(backend, "name", "unknown")
    return value() if callable(value) else str(value)


def run_ibm_estimator(
    circuits: Sequence[object],
    mapping: Sequence[Tuple[int, int]],
    cfg: TemporalIsingQRCConfig,
    params: ReservoirParameters,
    shots: int = 1024,
    backend_name: str = "auto",
    optimization_level: int = 3,
    resilience_level: int = 1,
    dynamical_decoupling: bool = True,
    dd_sequence: str = "XY4",
    max_pubs_per_job: int = 100,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Execute checkpoint circuits on IBM hardware through Qiskit Runtime EstimatorV2.

    Credentials are intentionally not accepted as command-line arguments. Configure them in
    the trusted qBraid/IBM environment or through QiskitRuntimeService environment variables.
    """
    try:
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import EstimatorOptions, EstimatorV2 as Estimator
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError(
            "IBM execution requires qiskit>=2.4 and qiskit-ibm-runtime>=0.46."
        ) from exc

    service = QiskitRuntimeService()
    if backend_name == "auto":
        backend = service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=cfg.n_qubits,
        )
    else:
        backend = service.backend(backend_name)

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=int(optimization_level),
    )
    labels, observables = qrc_observables(cfg, params)
    isa_circuits = [pass_manager.run(circuit) for circuit in circuits]
    pubs = []
    transpiled_depths, transpiled_sizes, transpiled_twoq = [], [], []
    for circuit in isa_circuits:
        mapped = [observable.apply_layout(circuit.layout) for observable in observables]
        pubs.append((circuit, mapped))
        transpiled_depths.append(int(circuit.depth() or 0))
        transpiled_sizes.append(int(circuit.size()))
        ops = circuit.count_ops()
        transpiled_twoq.append(
            int(sum(v for k, v in ops.items() if k in {"cx", "cz", "ecr", "rzz", "rxx", "ryy"}))
        )

    options = EstimatorOptions()
    options.default_shots = int(shots)
    options.resilience_level = int(resilience_level)
    if dynamical_decoupling:
        options.dynamical_decoupling.enable = True
        options.dynamical_decoupling.sequence_type = dd_sequence

    estimator = Estimator(mode=backend, options=options)
    vectors: List[np.ndarray] = []
    jobs: List[Dict[str, object]] = []
    started = time.time()
    for start in range(0, len(pubs), max_pubs_per_job):
        chunk = pubs[start : start + max_pubs_per_job]
        job = estimator.run(chunk)
        job_id = job.job_id()
        result = job.result()
        for pub_result in result:
            vectors.append(expectation_vector_to_features(pub_result.data.evs, cfg, params))
        record: Dict[str, object] = {
            "job_id": job_id,
            "pub_start": start,
            "pub_count": len(chunk),
        }
        try:
            record["metrics"] = job.metrics()
        except Exception:
            pass
        jobs.append(record)

    metadata: Dict[str, object] = {
        "provider": "IBM Quantum Platform",
        "backend": _backend_name(backend),
        "num_qubits_backend": int(getattr(backend, "num_qubits", 0)),
        "qrc_qubits": cfg.n_qubits,
        "shots": int(shots),
        "optimization_level": int(optimization_level),
        "resilience_level": int(resilience_level),
        "dynamical_decoupling": bool(dynamical_decoupling),
        "dd_sequence": dd_sequence if dynamical_decoupling else None,
        "observable_labels": labels,
        "job_records": jobs,
        "wall_clock_sec": float(time.time() - started),
        "transpiled_depth_min": int(min(transpiled_depths)) if transpiled_depths else 0,
        "transpiled_depth_mean": float(np.mean(transpiled_depths)) if transpiled_depths else 0.0,
        "transpiled_depth_max": int(max(transpiled_depths)) if transpiled_depths else 0,
        "transpiled_gate_count_mean": float(np.mean(transpiled_sizes)) if transpiled_sizes else 0.0,
        "transpiled_two_qubit_gates_mean": float(np.mean(transpiled_twoq)) if transpiled_twoq else 0.0,
    }
    try:
        metadata["active_instance"] = service.active_instance()
    except Exception:
        pass
    return assemble_feature_tensor(vectors, mapping), metadata


def run_qbraid_managed_counts(
    measured_circuits: Sequence[object],
    mapping: Sequence[Tuple[int, int]],
    cfg: TemporalIsingQRCConfig,
    params: ReservoirParameters,
    device_qrn: str,
    shots: int = 1024,
    max_circuits_per_batch: int = 25,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Execute on a qBraid-managed non-IBM QPU and derive Z/ZZ features from counts."""
    try:
        # qBraid SDK v0.12+ exports QbraidProvider at package level.
        from qbraid import QbraidProvider
    except ImportError:
        try:
            # Backward-compatible fallback for older qBraid images.
            from qbraid.runtime import QbraidProvider
        except ImportError as exc:
            raise RuntimeError("Install qbraid>=0.12 to use managed qBraid devices") from exc

    provider = QbraidProvider()
    device = provider.get_device(device_qrn)
    vectors: List[np.ndarray] = []
    job_records: List[Dict[str, object]] = []
    started = time.time()

    for start in range(0, len(measured_circuits), max_circuits_per_batch):
        chunk = list(measured_circuits[start : start + max_circuits_per_batch])
        try:
            submitted = device.run(chunk, shots=int(shots))
            jobs = list(submitted) if isinstance(submitted, (list, tuple)) else [submitted]
        except Exception:
            # Conservative fallback for providers that only accept one program per job.
            jobs = [device.run(circuit, shots=int(shots)) for circuit in chunk]

        for local_idx, job in enumerate(jobs):
            try:
                job.wait_for_final_state()
            except Exception:
                pass
            result = job.result()
            counts = result.data.get_counts()
            vectors.append(counts_to_features(counts, cfg, params))
            raw_id = getattr(job, "id", "unknown")
            job_id = raw_id() if callable(raw_id) else raw_id
            job_records.append(
                {
                    "job_id": str(job_id),
                    "global_circuit_index": start + local_idx,
                }
            )

    raw_metadata = getattr(device, "metadata", {})
    device_metadata = raw_metadata() if callable(raw_metadata) else raw_metadata
    meta = {
        "provider": "qBraid managed access",
        "device_qrn": device_qrn,
        "device_metadata": device_metadata,
        "shots": int(shots),
        "jobs": job_records,
        "wall_clock_sec": float(time.time() - started),
    }
    return assemble_feature_tensor(vectors, mapping), meta
