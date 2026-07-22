import numpy as np

from src.phase3.metrics import classification_metrics, regression_metrics
from src.phase3.qrc_parameters import TemporalIsingQRCConfig, feature_dimension, topology_edges
from src.phase3.temporal_ising_qrc import encode_exact


def test_metrics_are_finite():
    m = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.1]))
    assert np.isfinite(m["qlike"])
    c = classification_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert c["macro_f1"] == 1.0


def test_exact_qrc_shape_and_determinism():
    cfg = TemporalIsingQRCConfig(n_qubits=3, temporal_bins=4, virtual_nodes=1, topology="ring")
    X = np.linspace(-1, 1, 40, dtype=np.float32).reshape(10, 4, 1)
    a, meta = encode_exact(X, cfg, seed=2, batch_size=5, device="cpu")
    b, _ = encode_exact(X, cfg, seed=2, batch_size=2, device="cpu")
    assert a.shape == (10, 4, feature_dimension(cfg, topology_edges(3, "ring")))
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert meta["trainable_quantum_parameters"] == 0
