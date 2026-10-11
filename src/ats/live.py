from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request
from src.ats.resilience import fetch_json


@dataclass(frozen=True)
class LiveQuote:
    instrument_key: str
    price: float
    prev_close: float | None
    open: float | None
    high: float | None
    low: float | None
    volume: float | None
    event_time: str
    source: str


class UpstoxLiveProvider:
    """Low-overhead Upstox V3 LTP adapter with a stable ATS quote contract."""

    URL = "https://api.upstox.com/v3/market-quote/ltp"
    FULL_QUOTE_URL = "https://api.upstox.com/v3/market-quote/quotes"

    def __init__(self, access_token: str | None = None, instrument_key: str | None = None):
        self.access_token = access_token or os.getenv("ATS_UPSTOX_ACCESS_TOKEN")
        self.instrument_key = instrument_key or os.getenv("ATS_UPSTOX_NIFTY_KEY", "NSE_INDEX|Nifty 50")

    @property
    def configured(self) -> bool:
        return bool(self.access_token)

    def fetch_full_quote(self) -> dict:
        """Return a genuine exchange-backed top-5 market-depth snapshot.

        WHY: live execution must use observed bid/ask depth, never a fabricated spread.
        WATCH: the configured instrument must be tradeable; index instruments have no executable book.
        """
        if not self.access_token:
            raise RuntimeError("ATS_UPSTOX_ACCESS_TOKEN is not configured")
        req = Request(
            f"{self.FULL_QUOTE_URL}?instrument_key={quote(self.instrument_key, safe='')}",
            headers={"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"},
        )
        payload = fetch_json(req, timeout=8)
        data = payload.get("data", {})
        key = self.instrument_key.replace("|", ":")
        item = data.get(key) or data.get(self.instrument_key)
        if not item:
            raise RuntimeError(f"Upstox full quote did not contain {self.instrument_key}")
        depth = item.get("depth") or {}
        bids = tuple((float(x["price"]), int(x["quantity"])) for x in depth.get("buy", []) if x.get("price", 0) > 0 and x.get("quantity", 0) > 0)
        asks = tuple((float(x["price"]), int(x["quantity"])) for x in depth.get("sell", []) if x.get("price", 0) > 0 and x.get("quantity", 0) > 0)
        if not bids or not asks:
            raise RuntimeError("Upstox full quote returned no executable bid/ask depth")
        ts = item.get("timestamp") or payload.get("timestamp")
        if ts is None:
            raise RuntimeError("Depth response has no exchange timestamp")
        bids = tuple(sorted(bids, reverse=True))
        asks = tuple(sorted(asks))
        return {
            "instrument_key": self.instrument_key,
            "timestamp": str(ts) if ts is not None else datetime.now(timezone.utc).isoformat(),
            "last_price": float(item.get("last_price")),
            "bids": bids,
            "asks": asks,
            "ohlc": item.get("ohlc", {}),
            "source": "UpstoxV3-FullMarketQuote",
        }

    def fetch_nifty(self) -> LiveQuote:
        if not self.access_token:
            raise RuntimeError("ATS_UPSTOX_ACCESS_TOKEN is not configured")
        req = Request(
            f"{self.URL}?instrument_key={quote(self.instrument_key, safe='')}",
            headers={"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"},
        )
        payload = fetch_json(req, timeout=8)
        data = payload.get("data", {})
        item = data.get(self.instrument_key) or data.get(self.instrument_key.replace("|", ":"))
        if not item or "last_price" not in item:
            raise RuntimeError(f"Upstox response did not contain {self.instrument_key}")
        ts_ms = item.get("timestamp") or item.get("last_trade_time")
        event_time = (
            datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc).isoformat()
            if ts_ms
            else datetime.now(timezone.utc).isoformat()
        )
        prev_close = item.get("cp")
        price = float(item["last_price"])
        if prev_close is None:
            prev_close = price
        return LiveQuote(
            instrument_key=self.instrument_key,
            price=price,
            prev_close=float(prev_close),
            open=None,
            high=None,
            low=None,
            volume=float(item.get("volume", 0) or 0),
            event_time=event_time,
            source="UpstoxV3-LTP",
        )
    def fetch_nifty_intraday(self, interval_minutes: int = 5) -> list[dict]:
        """Fetch current-session NIFTY candles from Upstox V3.

        WHY: forecast features must come from the same trusted live source as the quote.
        WATCH: Upstox intraday data is current-session only; historical replay uses the
        separate V3 historical-candle endpoint.
        """
        if not self.access_token:
            raise RuntimeError("ATS_UPSTOX_ACCESS_TOKEN is not configured")
        if not 1 <= interval_minutes <= 300:
            raise ValueError("interval_minutes must be between 1 and 300")
        url = (
            "https://api.upstox.com/v3/historical-candle/intraday/"
            f"{quote(self.instrument_key, safe='')}/minutes/{interval_minutes}"
        )
        req = Request(
            url,
            headers={"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"},
        )
        payload = fetch_json(req, timeout=8)
        candles = payload.get("data", {}).get("candles", [])
        rows = []
        for c in candles:
            if len(c) < 6:
                continue
            rows.append({
                "event_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5] or 0),
                "source": "UpstoxV3-Intraday",
            })
        rows.sort(key=lambda r: r["event_time"])
        return rows

    def fetch_nifty_history(self, to_date: str, from_date: str, interval_minutes: int = 15) -> list[dict]:
        """Fetch bounded historical NIFTY candles from Upstox V3."""
        if not self.access_token:
            raise RuntimeError("ATS_UPSTOX_ACCESS_TOKEN is not configured")
        url = (
            "https://api.upstox.com/v3/historical-candle/"
            f"{quote(self.instrument_key, safe='')}/minutes/{interval_minutes}/{to_date}/{from_date}"
        )
        req = Request(
            url,
            headers={"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"},
        )
        payload = fetch_json(req, timeout=12)
        candles = payload.get("data", {}).get("candles", [])
        rows = []
        for c in candles:
            if len(c) < 6:
                continue
            rows.append({
                "event_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5] or 0),
                "source": "UpstoxV3-Historical",
            })
        rows.sort(key=lambda r: r["event_time"])
        return rows
