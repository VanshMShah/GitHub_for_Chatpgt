from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ats.market_session import MarketSession
from src.ats.models import Forecast
from src.ats.paper.live import LivePaperConfig, LivePaperRunner
from src.ats.store import Store


class FakeProvider:
    instrument_key = "NSE_EQ|TEST"


    def fetch_full_quote(self):
        return {
            "instrument_key": self.instrument_key,
            "timestamp": "2026-09-15T10:00:00+00:00",
            "last_price": 100.5,
            "bids": ((100.4, 50), (100.3, 100)),
            "asks": ((100.6, 50), (100.7, 100)),
            "ohlc": {},
            "source": "fake-full-depth",
        }

    def fetch_nifty_intraday(self, interval_minutes=1):
        return [{
            "event_time": "2026-09-14T10:00:00+00:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "source": "fake",
        }]


class FakePQRNN:
    def predict(self, frame):
        return Forecast(0.01, 0.80, -0.01, -0.002, 0.01, 0.02, 0.04, 0.02,
                        {"expert": "pqrnn_intraday", "horizon": "60s"})


class FakeEngine:
    def __init__(self, store):
        self.store = store
        self.live_provider = FakeProvider()
        self.pqrnn_intraday = FakePQRNN()

    def _load_intraday(self):
        rows = self.store.query(
            "SELECT event_time,open,high,low,close,volume FROM intraday_observations ORDER BY event_time"
        )
        return __import__("pandas").DataFrame(rows, columns=["event_time", "open", "high", "low", "close", "volume"])


def test_nse_market_calendar_rejects_weekend_and_accepts_session():
    cal = MarketSession()
    assert not cal.is_open(datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc))
    assert cal.is_open(datetime(2026, 9, 15, 5, 0, tzinfo=timezone.utc))


def test_live_paper_stays_disabled_by_default(tmp_path: Path):
    runner = LivePaperRunner(
        FakeEngine(Store(tmp_path / "ats.db")),
        FakeProvider(),
        LivePaperConfig(enable=False, audit_dir=str(tmp_path / "audit")),
        now_fn=lambda: datetime(2026, 9, 15, 5, 0, tzinfo=timezone.utc),
    )
    result = runner.tick()
    assert result["state"] == "DISABLED"


def test_kill_switch_halts_before_model_execution(tmp_path: Path):
    kill = tmp_path / "PAPER_KILL"
    kill.touch()
    runner = LivePaperRunner(
        FakeEngine(Store(tmp_path / "ats.db")),
        FakeProvider(),
        LivePaperConfig(enable=True, kill_switch_path=str(kill), audit_dir=str(tmp_path / "audit")),
        now_fn=lambda: datetime(2026, 9, 15, 5, 0, tzinfo=timezone.utc),
    )
    result = runner.tick()
    assert result["state"] == "HALTED_KILL_SWITCH"


def test_synthetic_bbo_is_explicit_and_local_fill_only(tmp_path: Path):
    runner = LivePaperRunner(
        FakeEngine(Store(tmp_path / "ats.db")),
        FakeProvider(),
        LivePaperConfig(enable=True, execution_instrument_key="NSE_EQ|TEST", audit_dir=str(tmp_path / "audit")),
        now_fn=lambda: datetime(2026, 9, 15, 5, 2, tzinfo=timezone.utc),
    )
    result = runner.tick()
    assert result["state"] in {"WARMUP", "NO_ORDER", "ORDER_NO_FILL", "FILL", "BLOCKED_NO_GENUINE_DEPTH"}
    rows = runner.engine.store.query("SELECT COUNT(*) FROM intraday_observations")
    assert rows[0][0] == 1


def test_market_calendar_fails_closed_for_unknown_year():
    cal = MarketSession()
    import pytest
    with pytest.raises(RuntimeError, match="calendar not provisioned"):
        cal.is_market_day(datetime(2027, 1, 4, 5, 0, tzinfo=timezone.utc))


def test_index_instrument_is_rejected_for_execution_depth(tmp_path: Path):
    provider = FakeProvider()
    provider.instrument_key = "NSE_INDEX|Nifty 50"
    runner = LivePaperRunner(
        FakeEngine(Store(tmp_path / "ats.db")),
        provider,
        LivePaperConfig(enable=True, execution_instrument_key=provider.instrument_key, audit_dir=str(tmp_path / "audit")),
        now_fn=lambda: datetime(2026, 9, 15, 5, 2, tzinfo=timezone.utc),
    )
    result = runner.tick()
    assert result["state"] == "BLOCKED_NO_GENUINE_DEPTH"
    assert "tradeable NSE instrument" in result["reason"]
