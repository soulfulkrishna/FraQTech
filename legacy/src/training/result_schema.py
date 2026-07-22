from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class ResultRow:
    dataset: str
    model: str
    encoder_type: str
    readout_type: str
    seed: int
    split_id: str

    rmse: float
    nrmse_sigma: float
    mae: float

    qrc_cache_time_sec: float
    final_train_time_sec: float
    hpo_time_sec: float
    inference_latency_ms_per_sample: float

    energy_kwh_cache: float
    energy_kwh_final_train: float
    energy_kwh_hpo: float
    energy_kwh_total: float

    carbon_kgco2e_cache: float
    carbon_kgco2e_final_train: float
    carbon_kgco2e_hpo: float
    carbon_kgco2e_total: float

    peak_ram_gb: float
    peak_gpu_mem_gb: float

    trainable_params: int
    total_params: int
    feature_dim: int

    qubits: Optional[int]
    modes: Optional[int]
    virtual_nodes: Optional[int]
    circuit_depth: Optional[int]
    circuit_evals: Optional[int]
    shots: Optional[int]
    qpu_time_proxy_sec: Optional[float]

    backend_type: str
    git_commit: str
    hardware_id: str

    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)