from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)

@dataclass(frozen=True)
class Settings:
    execution_key: str = os.getenv("ATS_UPSTOX_EXECUTION_KEY", "")
    model_dir: Path = Path(os.getenv("ATS_MODEL_DIR", "models"))
    api_token: str = os.getenv("ATS_API_TOKEN", "")
    daily_publication_lag_minutes: int = int(os.getenv("ATS_DAILY_PUBLICATION_LAG_MINUTES", "15"))
    db_path: Path = Path(os.getenv("ATS_DB_PATH", "data/ats_state.sqlite3"))
    market_poll_seconds: int = int(os.getenv("ATS_POLL_SECONDS_MARKET", "300"))
    offmarket_poll_seconds: int = int(os.getenv("ATS_POLL_SECONDS_OFFMARKET", "1800"))
    min_observations: int = int(os.getenv("ATS_MIN_OBSERVATIONS", "220"))
    lookback_days: int = int(os.getenv("ATS_LOOKBACK_DAYS", "756"))
    horizon_days: int = int(os.getenv("ATS_FORECAST_HORIZON_DAYS", "1"))
    live_poll_seconds: int = int(os.getenv("ATS_LIVE_POLL_SECONDS", "30"))
    auto_start_agent: bool = os.getenv("ATS_AUTOSTART", "true").lower() in {"1", "true", "yes", "on"}
    forecast_interval_market_seconds: int = int(os.getenv("ATS_FORECAST_INTERVAL_MARKET", "300"))
    forecast_interval_offmarket_seconds: int = int(os.getenv("ATS_FORECAST_INTERVAL_OFFMARKET", "3600"))
    upstox_access_token: str = os.getenv("ATS_UPSTOX_ACCESS_TOKEN", "")
    upstox_nifty_key: str = os.getenv("ATS_UPSTOX_NIFTY_KEY", "NSE_INDEX|Nifty 50")
    live_paper_enabled: bool = os.getenv("ATS_LIVE_PAPER_ENABLE", "false").lower() in {"1", "true", "yes", "on"}

settings = Settings()
if settings.min_observations < 60 or settings.horizon_days != 1:
    raise ValueError("ATS_MIN_OBSERVATIONS must be >=60; this release supports only a one-session daily horizon")
if min(settings.live_poll_seconds, settings.offmarket_poll_seconds) < 1:
    raise ValueError("Polling intervals must be positive")
settings.db_path.parent.mkdir(parents=True, exist_ok=True)
