from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ats.config import settings
from src.ats.engine import ATSEngine
from src.ats.live import UpstoxLiveProvider
from src.ats.paper.live import LivePaperConfig, LivePaperRunner
from src.ats.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ATS live-data, local-execution paper evaluation")
    parser.add_argument("--once", action="store_true", help="run one market tick and exit")
    args = parser.parse_args()

    cfg = LivePaperConfig(
        interval_seconds=int(__import__("os").getenv("ATS_LIVE_PAPER_INTERVAL", "60")),
        enable=__import__("os").getenv("ATS_LIVE_PAPER_ENABLE", "false").lower() in {"1", "true", "yes", "on"},
        execution_instrument_key=__import__("os").getenv("ATS_UPSTOX_EXECUTION_KEY", ""),
        kill_switch_path=__import__("os").getenv("ATS_PAPER_KILL_SWITCH", "data/PAPER_KILL"),
        lot_size=int(__import__("os").getenv("ATS_PAPER_LOT_SIZE", "1")),
        lot_size_confirmed=bool(__import__("os").getenv("ATS_PAPER_LOT_SIZE")),
        allow_short=__import__("os").getenv("ATS_PAPER_ALLOW_SHORT","false").lower()=="true",
        fee_bps=float(__import__("os").getenv("ATS_PAPER_FEE_BPS","0")),
        fixed_fee=float(__import__("os").getenv("ATS_PAPER_FIXED_FEE","0")),
        target_notional=float(__import__("os").getenv("ATS_PAPER_TARGET_NOTIONAL","1000")),
        max_notional=float(__import__("os").getenv("ATS_PAPER_MAX_NOTIONAL","10000")),
        max_order_notional=float(__import__("os").getenv("ATS_PAPER_MAX_ORDER_NOTIONAL","2000")),
        max_daily_loss=float(__import__("os").getenv("ATS_PAPER_MAX_DAILY_LOSS","1000")),
        audit_dir=__import__("os").getenv("ATS_LIVE_PAPER_AUDIT_DIR", "data/live_paper"),
    )
    if not settings.upstox_access_token:
        raise SystemExit("ATS_UPSTOX_ACCESS_TOKEN is required")

    store = Store(settings.db_path)
    engine = ATSEngine(store)
    execution_key = __import__("os").getenv("ATS_UPSTOX_EXECUTION_KEY", "")
    if not execution_key:
        raise SystemExit("ATS_UPSTOX_EXECUTION_KEY is required and must be a tradeable NSE_EQ| or NSE_FO| instrument")
    provider = UpstoxLiveProvider(settings.upstox_access_token, execution_key)
    runner = LivePaperRunner(engine, provider, cfg)

    try:
        while True:
            try:
                result = runner.tick()
                engine.settle()
            except Exception as exc:
                result = runner._heartbeat("ERROR",reason=str(exc))
            print(result, flush=True)
            if args.once:
                return
            time.sleep(cfg.interval_seconds)
    except KeyboardInterrupt:
        print("ATS live-data paper runner stopped")


if __name__ == "__main__":
    main()
