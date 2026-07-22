from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import yaml

from src.models_qrc.gaussian_qrc import GaussianQRC, GaussianQRCConfig
from src.models_qrc.gate_qrc import GateQRC, GateQRCConfig


def load_qrc_config(project_root: Path, config_name: str) -> Dict[str, Any]:
    path = project_root / "configs" / "models" / f"{config_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"QRC config not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["config_name"] = config_name
    return cfg


def _filter_dataclass_kwargs(dataclass_type, cfg: Dict[str, Any]) -> Dict[str, Any]:
    allowed = set(dataclass_type.__dataclass_fields__.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def build_qrc_encoder(
    cfg: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    encoder_type = cfg["encoder_type"]
    seed_offset = int(cfg.get("seed_offset", 0))
    qrc_seed = seed + seed_offset

    torch.manual_seed(qrc_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(qrc_seed)

    if encoder_type == "cv_gqrc":
        kwargs = _filter_dataclass_kwargs(GaussianQRCConfig, cfg)
        qrc_cfg = GaussianQRCConfig(**kwargs)
        encoder = GaussianQRC(qrc_cfg).to(device)

        meta = {
            "encoder_type": encoder_type,
            "backend_type": cfg.get("backend_type", "cv_moment_simulator"),
            "modes": qrc_cfg.modes,
            "virtual_nodes": qrc_cfg.virtual_nodes,
            "qubits": None,
            "washout": qrc_cfg.washout,
            "feature_dim": encoder.feature_dim,
            "total_params": encoder.total_params(),
            "trainable_params": encoder.trainable_params(),
            "circuit_depth": None,
            "shots": 0,
            "config": cfg,
            "dataclass_config": asdict(qrc_cfg),
        }

        return encoder, meta

    if encoder_type == "gb_qrc":
        kwargs = _filter_dataclass_kwargs(GateQRCConfig, cfg)
        qrc_cfg = GateQRCConfig(**kwargs)
        encoder = GateQRC(qrc_cfg, seed=qrc_seed).to(device)

        meta = {
            "encoder_type": encoder_type,
            "backend_type": cfg.get("backend_type", "gate_statevector_simulator"),
            "modes": None,
            "virtual_nodes": qrc_cfg.virtual_nodes,
            "qubits": qrc_cfg.n_qubits,
            "washout": qrc_cfg.washout,
            "feature_dim": encoder.feature_dim,
            "total_params": encoder.total_params(),
            "trainable_params": encoder.trainable_params(),
            "circuit_depth": encoder.circuit_depth_proxy(),
            "shots": qrc_cfg.shots,
            "config": cfg,
            "dataclass_config": asdict(qrc_cfg),
        }

        return encoder, meta

    raise ValueError(f"Unsupported encoder_type={encoder_type}")