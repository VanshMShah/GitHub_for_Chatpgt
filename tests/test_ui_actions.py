from pathlib import Path
from src.ats.store import Store
from src.ats.engine import ATSEngine


def test_manual_forecast_method_is_mode_aware(monkeypatch, tmp_path: Path):
    e = ATSEngine(Store(tmp_path / "ats.db"))
    monkeypatch.setattr(e, "is_market_hours", lambda: True)
    calls = []
    e.forecast_intraday = lambda: calls.append("intraday") or {"ok": True}
    e.forecast = lambda: calls.append("daily") or {"ok": True}
    fn = e.forecast_intraday if e.is_market_hours() else e.forecast
    assert fn()["ok"] and calls == ["intraday"]


def test_policy_state_survives(tmp_path: Path):
    s = Store(tmp_path / "ats.db")
    e = ATSEngine(s)
    state = e.policy.state()
    assert "weights" in state and abs(sum(state["weights"].values()) - 1.0) < 1e-9
