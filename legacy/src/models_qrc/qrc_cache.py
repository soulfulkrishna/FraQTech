import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

from src.models_qrc.cv_gqrc import CVGaussianQRC


def _hash_dict(d: Dict) -> str:
    text = json.dumps(d, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_qrc_config(project_root: Path, encoder: str) -> Dict:
    config_path = project_root / "configs" / "models" / f"{encoder}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"QRC config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_cv_gqrc(config: Dict, seed: int) -> CVGaussianQRC:
    seed_offset = int(config.get("seed_offset", 10000))

    return CVGaussianQRC(
        modes=int(config["modes"]),
        virtual_nodes=int(config["virtual_nodes"]),
        washout=int(config["washout"]),
        spectral_radius=float(config["spectral_radius"]),
        leak_rate=float(config["leak_rate"]),
        input_scale=float(config["input_scale"]),
        noise_scale=float(config["noise_scale"]),
        include_means=bool(config["include_means"]),
        include_variances=bool(config["include_variances"]),
        include_covariances=bool(config["include_covariances"]),
        include_squared_means=bool(config["include_squared_means"]),
        include_abs_means=bool(config["include_abs_means"]),
        seed=seed + seed_offset,
    )


def get_qrc_cache_paths(
    project_root: Path,
    dataset: str,
    encoder: str,
    seed: int,
    config: Dict,
) -> Tuple[Path, Path]:
    cache_dir = project_root / "data" / "qrc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_hash = _hash_dict(config)

    feature_path = cache_dir / f"{dataset}_{encoder}_seed{seed}_{config_hash}_features.npz"
    meta_path = cache_dir / f"{dataset}_{encoder}_seed{seed}_{config_hash}_meta.json"

    return feature_path, meta_path


def save_qrc_cache(
    feature_path: Path,
    meta_path: Path,
    train_val_features: np.ndarray,
    test_features: np.ndarray,
    meta: Dict,
) -> None:
    np.savez_compressed(
        feature_path,
        train_val_features=train_val_features,
        test_features=test_features,
    )

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_qrc_cache(feature_path: Path, meta_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    data = np.load(feature_path)
    train_val_features = data["train_val_features"]
    test_features = data["test_features"]

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return train_val_features, test_features, meta