import pytest
pytest.importorskip("torch", reason="Optional neural tests require torch")
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ats.pqrnn_intraday import IntradayPQRNNConfig, IntradayPQRNNLifecycle, make_intraday_frame


def bars(n=700, step_seconds=60):
    rng = np.random.default_rng(9)
    ts = pd.date_range("2026-01-01", periods=n, freq=f"{step_seconds}s", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    return pd.DataFrame({
        "event_time": ts,
        "open": close,
        "high": close * 1.0005,
        "low": close * 0.9995,
        "close": close,
        "volume": rng.integers(100, 1000, n),
    })


def test_target_uses_only_future_close_and_is_not_feature_leakage():
    raw = make_intraday_frame(bars())
    supervised, cols = IntradayPQRNNLifecycle.make_supervised(raw, 1)
    assert "target_ret" not in cols
    row = supervised.iloc[100]
    source = raw.loc[raw["event_time"] == row["event_time"]].iloc[0]
    future = raw.loc[raw["event_time"] == row["target_end_time"]].iloc[0]
    expected = np.log(future["close"] / source["close"])
    assert row["target_ret"] == pytest.approx(expected)
    assert row["target_end_time"] > row["event_time"]


def test_training_scaler_does_not_use_holdout_rows(tmp_path: Path):
    raw = make_intraday_frame(bars())
    # Perturb only the final holdout region dramatically.
    altered = raw.copy()
    altered.loc[600:, "ret"] = altered.loc[600:, "ret"] + 100.0
    altered.loc[600:, "ret3"] = altered.loc[600:, "ret3"] + 100.0
    altered.loc[600:, "ret12"] = altered.loc[600:, "ret12"] + 100.0
    altered.loc[600:, "rv12"] = altered.loc[600:, "rv12"] + 100.0
    altered.loc[600:, "rv48"] = altered.loc[600:, "rv48"] + 100.0
    altered.loc[600:, "range"] = altered.loc[600:, "range"] + 100.0
    altered.loc[600:, "volume_z"] = altered.loc[600:, "volume_z"] + 100.0
    lifecycle = IntradayPQRNNLifecycle.__new__(IntradayPQRNNLifecycle)
    lifecycle.cfg = IntradayPQRNNConfig(min_rows=500, holdout_rows=100)
    a = lifecycle._sequences(raw)
    b = lifecycle._sequences(altered)
    np.testing.assert_allclose(a[4], b[4])
    np.testing.assert_allclose(a[5], b[5])


def test_resolution_guard_rejects_15m_for_60s_model(tmp_path: Path):
    class Stub:
        pass
    lifecycle = IntradayPQRNNLifecycle.__new__(IntradayPQRNNLifecycle)
    lifecycle.cfg = IntradayPQRNNConfig()
    with pytest.raises(RuntimeError, match="approximately 60-second bars"):
        lifecycle.validate_resolution(bars(100, step_seconds=900))
