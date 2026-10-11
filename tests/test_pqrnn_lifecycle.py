import pytest
pytest.importorskip("torch", reason="Optional neural tests require torch")
import numpy as np

from src.ats.pqrnn_lifecycle import PQRNNLifecycle, PQRNNLifecycleConfig


def test_block_bootstrap_wider_than_iid_on_autocorrelated_series():
    """Block bootstrap CI must be wider on strongly autocorrelated data."""
    rng = np.random.default_rng(11)
    n = 200
    eps = rng.normal(0, 1, n)
    d = np.zeros(n)
    for i in range(1, n):
        d[i] = 0.8 * d[i - 1] + eps[i]
    d -= d.mean()

    lifecycle = PQRNNLifecycle.__new__(PQRNNLifecycle)
    lifecycle.cfg = PQRNNLifecycleConfig(seed=11, bootstrap_reps=300, block_length=5)

    _, lo_block, hi_block = lifecycle._bootstrap_pvalue(d, np.zeros(n))
    _, lo_iid, hi_iid = lifecycle._bootstrap_pvalue_iid_reference(d, np.zeros(n))

    assert (hi_block - lo_block) > (hi_iid - lo_iid)


def test_checkpoint_selection_uses_validation_not_training_loss():
    """Checkpoint selection must use an internal validation slice and may stop before the last epoch."""
    rng = np.random.default_rng(123)
    x = rng.normal(size=(100, 20, 6)).astype(np.float32)
    y = (0.3 * x[:, 0, 0] - 0.2 * x[:, 1, 1] + rng.normal(0, 1, 100)).astype(np.float32)

    lifecycle = PQRNNLifecycle.__new__(PQRNNLifecycle)
    lifecycle.cfg = PQRNNLifecycleConfig(
        seed=7,
        train_epochs=20,
        batch_size=16,
        learning_rate=5e-2,
    )
    lifecycle.device = "cpu"

    lifecycle._train_candidate(x, y, split=80)

    assert lifecycle._last_checkpoint_used_validation is True
    assert 0 <= lifecycle._last_checkpoint_epoch < lifecycle.cfg.train_epochs - 1
