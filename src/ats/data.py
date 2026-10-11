from __future__ import annotations
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request
from src.ats.resilience import fetch_json
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Bar:
    event_time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str

class YahooChartProvider:
    """Dependency-free public chart adapter. Use a broker adapter for production-grade feeds."""
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
    def fetch(self, symbol: str, period: str = "2y", interval: str = "1d") -> list[Bar]:
        url = self.BASE.format(quote(symbol, safe="")) + f"?range={period}&interval={interval}&events=history"
        req = Request(url, headers={"User-Agent": "ATS-v2/1.0"})
        payload = fetch_json(req, timeout=15)
        result = payload["chart"]["result"][0]
        q = result["indicators"]["quote"][0]
        ts = result["timestamp"]
        bars: list[Bar] = []
        for i, epoch in enumerate(ts):
            vals = {k: q.get(k, [None] * len(ts))[i] for k in ("open","high","low","close","volume")}
            if vals["close"] is None:
                continue
            dt = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
            bars.append(Bar(dt, float(vals["open"]), float(vals["high"]), float(vals["low"]), float(vals["close"]), float(vals["volume"] or 0), "YahooChart"))
        return bars

def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    frame = pd.DataFrame([b.__dict__ for b in bars])
    if frame.empty:
        return frame
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    return frame.sort_values("event_time").drop_duplicates("event_time").reset_index(drop=True)
