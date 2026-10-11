from __future__ import annotations

import hashlib
import json
import math
import threading
from filelock import FileLock
import joblib
from functools import wraps
from src.ats.market_session import MarketSession

def serialized(fn):
    @wraps(fn)
    def call(self, *args, **kwargs):
        with self._operation_lock, self._process_lock:
            return fn(self, *args, **kwargs)
    return call

from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.ats.adaptation import diagnose, ResidualLearner
from src.ats.config import settings
from src.ats.data import YahooChartProvider
from src.ats.events import GDELTEventProvider
from src.ats.features import FEATURES, SYMBOLS, clean_training_frame, make_daily_features, make_intraday_features
from src.ats.live import UpstoxLiveProvider
from src.ats.models import ARIMAGARCHExpert, Forecast, LogisticExpert, PersistenceExpert, QuantileGBDT, blend
from src.ats.policy import DMAPolicy
from src.ats.risk import RiskEngine
from src.ats.pqrnn_lifecycle import PQRNNLifecycle
from src.ats.pqrnn_intraday import IntradayPQRNNConfig, IntradayPQRNNLifecycle, make_intraday_frame

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")


class ATSEngine:
    def __init__(self, store):
        self.store = store
        self._operation_lock = threading.RLock()
        self._process_lock = FileLock(str(store.path)+".workflow.lock",timeout=600)
        self.daily_artifact = settings.model_dir / "daily_experts.joblib"
        self.model_revision = None
        self.calendar = MarketSession()
        self.intraday_symbol = settings.execution_key or SYMBOLS["nifty"]
        self.provider = YahooChartProvider()
        self.live_provider = UpstoxLiveProvider(settings.upstox_access_token, settings.upstox_nifty_key)
        self.events_provider = GDELTEventProvider()
        self.policy = DMAPolicy(self.store.get_state("policy", {}))
        self.risk = RiskEngine(self.store)
        self.models: dict[str, object] = {}
        self._restore_models()
        self.intraday_model: LogisticExpert | None = None
        self.pqrnn_intraday = IntradayPQRNNLifecycle(self, self.store, IntradayPQRNNConfig())
        residual_state = self.store.get_state("residual_learner", {})
        self.residual = ResidualLearner(6, state=residual_state)
        self.pqrnn = PQRNNLifecycle(self, self.store)
        if self.pqrnn.champion_path.exists():
            self.policy.register_model("pqrnn", prior=0.10)
        self.store.set_state("policy", self.policy.state(), datetime.now(UTC).isoformat())

    def now_ist(self) -> datetime:
        return datetime.now(IST)

    def is_market_hours(self) -> bool:
        now = self.now_ist()
        return self.calendar.is_open(now)

    def _set_feed_health(self, name: str, ok: bool, source: str, error: str | None = None):
        now = datetime.now(UTC).isoformat()
        self.store.set_state(f"feed:{name}", {"ok": ok, "source": source, "updated_at": now, "error": error}, now)

    @serialized
    def ingest(self, period="2y"):
        counts = {}
        now = datetime.now(UTC).isoformat()
        for name, symbol in SYMBOLS.items():
            try:
                bars = self.provider.fetch(symbol, period=period, interval="1d")
                self._set_feed_health(name, True, "YahooChart")
            except Exception as exc:
                self._set_feed_health(name, False, "YahooChart", str(exc))
                continue
            if not bars:
                continue
            if name == "nifty":
                now_local = self.now_ist()
                bars = [b for b in bars if pd.Timestamp(b.event_time).date() < now_local.date() or
                        (pd.Timestamp(b.event_time).date() == now_local.date() and self.calendar.is_market_day(now_local) and
                         self.calendar.close_at(now_local.date()) + pd.Timedelta(minutes=settings.daily_publication_lag_minutes) <= now_local)]
                if not bars:
                    continue
            if name == "nifty":
                rows = [(symbol, b.event_time, now, b.open, b.high, b.low, b.close, b.volume, b.source) for b in bars]
                self.store.executemany("INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?,?,?)", rows)
            else:
                rows = [(symbol, b.event_time, now, b.close, b.source) for b in bars]
                self.store.executemany("INSERT OR REPLACE INTO exogenous VALUES(?,?,?,?,?)", rows)
            counts[name] = len(bars)
        self.store.set_state("last_ingest", now, now)
        return counts

    @serialized
    def ingest_live(self):
        """Use Upstox V3 when configured. Yahoo is an explicitly labelled fallback."""
        now = datetime.now(UTC).isoformat()
        if self.live_provider.configured:
            try:
                q = self.live_provider.fetch_nifty()
                # WHY: live snapshots must not contaminate daily training bars.
                self.store.execute(
                    "INSERT OR REPLACE INTO live_quotes(symbol,event_time,available_time,price,volume,source) VALUES(?,?,?,?,?,?)",
                    (SYMBOLS["nifty"], q.event_time, now, q.price, q.volume or 0.0, q.source),
                )
                self.store.set_state("live_quote", {
                    "price": q.price,
                    "event_time": q.event_time,
                    "source": q.source,
                    "prev_close": q.prev_close,
                }, now)
                self._set_feed_health("nifty_live", True, q.source)
                return {"price": q.price, "event_time": q.event_time, "source": q.source}
            except Exception as exc:
                self._set_feed_health("nifty_live", False, "UpstoxV3-LTP", str(exc))

        try:
            bars = self.provider.fetch("^NSEI", period="1d", interval="1m")
            if bars:
                b = bars[-1]
                self.store.execute(
                    "INSERT OR REPLACE INTO live_quotes(symbol,event_time,available_time,price,volume,source) VALUES(?,?,?,?,?,?)",
                    ("^NSEI", b.event_time, now, b.close, b.volume, "YahooChartFallback"),
                )
                self.store.set_state("live_quote", {
                    "price": b.close,
                    "event_time": b.event_time,
                    "source": "YahooChartFallback",
                    "prev_close": None,
                }, now)
                self._set_feed_health("nifty_live", True, "YahooChartFallback")
                return {"price": b.close, "event_time": b.event_time, "source": "YahooChartFallback"}
        except Exception as exc:
            self._set_feed_health("nifty_live", False, "YahooChartFallback", str(exc))
        return None

    @serialized
    def ingest_intraday(self, period="10d", interval="15m"):
        counts = {}
        now = datetime.now(UTC).isoformat()
        min_rows = 200

        # WHY: when a broker feed is configured, use it for the NIFTY forecast path
        # rather than mixing a live quote with a delayed historical forecast feed.
        if self.live_provider.configured:
            try:
                minutes = int(interval.replace("m", ""))
                rows_raw = self.live_provider.fetch_nifty_intraday(minutes)
                rows = [
                    (SYMBOLS["nifty"], r["event_time"], now, r["open"], r["high"], r["low"], r["close"], r["volume"], r["source"])
                    for r in rows_raw
                ]
                if rows:
                    self.store.executemany("INSERT OR REPLACE INTO intraday_observations VALUES(?,?,?,?,?,?,?,?,?)", rows)
                cached = self.store.query("SELECT COUNT(*) FROM intraday_observations WHERE symbol=?", (SYMBOLS["nifty"],))[0][0]
                counts["nifty"] = cached
                self._set_feed_health("intraday:nifty", cached >= min_rows, "UpstoxV3-Intraday", None if cached >= min_rows else f"Need {min_rows} intraday rows; have {cached}")
            except Exception as exc:
                self._set_feed_health("intraday:nifty", False, "UpstoxV3-Intraday", str(exc))
        else:
            # Yahoo's shorter intraday ranges can return fewer than the 200 usable rows
            # required by the feature set. Backfill progressively before declaring a failure.
            last_error = None
            successful_source = None
            for candidate_period in (period, "20d", "30d"):
                try:
                    bars = self.provider.fetch(SYMBOLS["nifty"], period=candidate_period, interval=interval)
                    rows = [(SYMBOLS["nifty"], b.event_time, now, b.open, b.high, b.low, b.close, b.volume, "YahooChartFallback") for b in bars]
                    if rows:
                        self.store.executemany("INSERT OR REPLACE INTO intraday_observations VALUES(?,?,?,?,?,?,?,?,?)", rows)
                    cached = self.store.query("SELECT COUNT(*) FROM intraday_observations WHERE symbol=?", (SYMBOLS["nifty"],))[0][0]
                    if cached >= min_rows:
                        successful_source = f"YahooChartFallback-{candidate_period}"
                        counts["nifty"] = cached
                        self._set_feed_health("intraday:nifty", True, successful_source)
                        break
                    last_error = RuntimeError(f"Need {min_rows} intraday rows; have {cached} after {candidate_period}")
                except Exception as exc:
                    last_error = exc
            if successful_source is None:
                cached = self.store.query("SELECT COUNT(*) FROM intraday_observations WHERE symbol=?", (SYMBOLS["nifty"],))[0][0]
                counts["nifty"] = cached
                # Keep cached history usable even when the newest provider request fails.
                self._set_feed_health("intraday:nifty", cached >= min_rows, "YahooChartFallback-CACHED", None if cached >= min_rows else str(last_error))

        # India VIX remains a separate series and falls back to Yahoo until a dedicated
        # broker/options feed is configured.
        try:
            bars = self.provider.fetch(SYMBOLS["india_vix"], period=period, interval=interval)
            rows = [(SYMBOLS["india_vix"], b.event_time, now, b.open, b.high, b.low, b.close, b.volume, b.source) for b in bars]
            if rows:
                self.store.executemany("INSERT OR REPLACE INTO intraday_observations VALUES(?,?,?,?,?,?,?,?,?)", rows)
            counts["india_vix"] = len(rows)
            self._set_feed_health("intraday:india_vix", bool(rows), "YahooChart")
        except Exception as exc:
            self._set_feed_health("intraday:india_vix", False, "YahooChart", str(exc))

        self.store.set_state("last_intraday_ingest", now, now)
        return counts

    @serialized
    def auto_adapt(self) -> dict:
        """Run evidence-gated PQRNN lifecycle off-market with bounded redundancy."""
        if self.is_market_hours():
            return {"status": "deferred", "reason": "market_hours"}
        last_error = None
        for attempt in range(1, 4):
            try:
                result = self.pqrnn.train_and_evaluate(force=False)
                self.store.set_state("pqrnn_auto_health", {"ok": True, "attempt": attempt, "updated_at": datetime.now(UTC).isoformat()}, datetime.now(UTC).isoformat())
                return result
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    import time
                    time.sleep(min(15.0, 1.0 * (2 ** (attempt - 1))))
        stamp = datetime.now(UTC).isoformat()
        failure = {"status": "retry_exhausted", "error": str(last_error), "updated_at": stamp}
        self.store.set_state("pqrnn_auto_health", failure, stamp)
        self.store.execute(
            "INSERT INTO diagnostics(forecast_id,timestamp,code,detail_json,action) VALUES(?,?,?,?,?)",
            (None, stamp, "PQRNN_AUTO_RETRY_EXHAUSTED", json.dumps(failure), "Keep incumbent and retry on next background cycle"),
        )
        return failure

    @serialized
    def sync(self, generate_forecast: bool = True) -> dict:
        """Incremental catch-up used by the UI and startup recovery.

        WHY: one click should reconcile only what is stale, not redownload every dataset blindly.
        """
        now = datetime.now(UTC).timestamp()
        result = {"daily": {}, "live": None, "intraday": {}, "events": 0}
        last_daily = float(self.store.get_state("last_daily_sync_epoch", 0) or 0)
        if now - last_daily > 1800 or self.store.query("SELECT COUNT(*) FROM observations")[0][0] == 0:
            result["daily"] = self.ingest("2y")
            self.store.set_state("last_daily_sync_epoch", now, datetime.now(UTC).isoformat())
            self.mark_retrain_pending()
        if self.is_market_hours() or self.store.get_state("latest_intraday_forecast") is None:
            result["live"] = self.ingest_live()
            if settings.execution_key and settings.upstox_access_token:
                try:
                    result["intraday"] = self.prepare_intraday()
                except Exception as exc:
                    self._set_feed_health("intraday", False, "Upstox", str(exc))
        last_evt = float(self.store.get_state("last_event_sync_epoch", 0) or 0)
        if now - last_evt > (900 if self.is_market_hours() else 1800):
            result["events"] = self.ingest_events()
            self.store.set_state("last_event_sync_epoch", now, datetime.now(UTC).isoformat())
        result["settled"] = self.settle()
        result["pqrnn"] = self.auto_adapt()
        if self.needs_retrain():
            result["train"] = self.train()
        if generate_forecast:
            result["forecast"] = self.forecast()
        return result

    @serialized
    def ingest_events(self, query="India OR NIFTY OR oil OR OPEC OR Iran OR China OR tariff OR RBI OR Fed", maxrecords=40):
        now = datetime.now(UTC).isoformat()
        try:
            rows = self.events_provider.fetch(query=query, maxrecords=maxrecords)
            if rows:
                self.store.executemany("INSERT OR REPLACE INTO events VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
            self._set_feed_health("events", True, "GDELT")
            self.store.set_state("last_event_ingest", now, now)
            return len(rows)
        except Exception as exc:
            self._set_feed_health("events", False, "GDELT", str(exc))
            return 0

    def _load(self):
        rows = self.store.query("SELECT event_time,open,high,low,close,volume FROM observations WHERE symbol=? ORDER BY event_time", (SYMBOLS["nifty"],))
        n = pd.DataFrame(rows, columns=["event_time", "open", "high", "low", "close", "volume"])
        if n.empty:
            return n, {}
        n["event_time"] = pd.to_datetime(n["event_time"], utc=True)
        exo = {}
        for name, symbol in SYMBOLS.items():
            if name == "nifty":
                continue
            rows = self.store.query("SELECT event_time,close FROM exogenous WHERE symbol=? ORDER BY event_time", (symbol,))
            f = pd.DataFrame(rows, columns=["event_time", "close"])
            if not f.empty:
                f["event_time"] = pd.to_datetime(f["event_time"], utc=True)
            exo[name] = f
        return n, exo

    def _load_intraday(self, symbol=None, interval_seconds=60):
        symbol = symbol or self.intraday_symbol
        rows = self.store.query("SELECT event_time,open,high,low,close,volume FROM instrument_bars WHERE symbol=? AND interval_seconds=? ORDER BY event_time", (symbol, interval_seconds))
        # Legacy index data is allowed only for migration/testing; the model validates resolution.
        if not rows and symbol == SYMBOLS["nifty"]:
            rows = self.store.query("SELECT event_time,open,high,low,close,volume FROM intraday_observations WHERE symbol=? ORDER BY event_time", (symbol,))
        f = pd.DataFrame(rows, columns=["event_time", "open", "high", "low", "close", "volume"])
        if not f.empty:
            f["event_time"] = pd.to_datetime(f["event_time"], utc=True)
        f.attrs["instrument_key"] = symbol
        f.attrs["interval_seconds"] = interval_seconds
        return f

    @serialized
    def prepare_intraday(self, days=7):
        if not settings.execution_key or not settings.upstox_access_token:
            raise RuntimeError("Set ATS_UPSTOX_EXECUTION_KEY and ATS_UPSTOX_ACCESS_TOKEN in .env first")
        if not settings.execution_key.startswith(("NSE_EQ|", "NSE_FO|")):
            raise RuntimeError("Execution requires an NSE_EQ or NSE_FO instrument")
        self.intraday_symbol = settings.execution_key
        provider = UpstoxLiveProvider(settings.upstox_access_token, self.intraday_symbol)
        now = self.now_ist()
        raw = provider.fetch_nifty_history(now.date().isoformat(), (now.date()-pd.Timedelta(days=days)).isoformat(), 1)
        raw += provider.fetch_nifty_intraday(1)
        rows = []
        for b in raw:
            ts = pd.Timestamp(b["event_time"])
            if ts.tzinfo is None:
                raise RuntimeError("Intraday provider timestamp must include timezone")
            ts = ts.tz_convert("UTC")
            if ts + pd.Timedelta(seconds=60) > pd.Timestamp(now):
                continue
            vals = [b[k] for k in ("open","high","low","close","volume")]
            if not np.isfinite(vals).all() or min(vals[:4]) <= 0 or vals[4] < 0:
                raise RuntimeError("Invalid intraday bar")
            rows.append((self.intraday_symbol,60,ts.isoformat(),datetime.now(UTC).isoformat(),*vals,b["source"]))
        self.store.executemany("INSERT OR REPLACE INTO instrument_bars VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
        count = len(self._load_intraday(self.intraday_symbol))
        return {"instrument":self.intraday_symbol,"interval_seconds":60,"rows":count}


    def needs_retrain(self) -> bool:
        self._restore_models()
        if not self.models:
            return True
        n, _ = self._load()
        if n.empty:
            return False
        trained_end = self.store.get_state("training_end")
        f = clean_training_frame(make_daily_features(*self._load()))
        latest = f["event_time"].max().isoformat() if not f.empty else None
        return trained_end != latest

    def mark_retrain_pending(self):
        self.store.set_state("training_pending", True, datetime.now(UTC).isoformat())

    @serialized
    def train(self):
        n, e = self._load()
        f = clean_training_frame(make_daily_features(n, e))
        if len(f) < settings.min_observations:
            raise RuntimeError(f"Need {settings.min_observations} usable daily observations; have {len(f)}")
        cols = [c for c in FEATURES if c in f.columns]
        X = f[cols].to_numpy()
        y = f["target_ret"].to_numpy()
        self.models = {
            "persistence": PersistenceExpert().fit(y),
            "logistic": LogisticExpert().fit(X, y),
            "lightgbm_compatible": QuantileGBDT().fit(X, y),
            "arima_gjr_garch": ARIMAGARCHExpert().fit(y[-min(len(y), settings.lookback_days):]),
        }
        now = datetime.now(UTC).isoformat()
        self.store.set_state("feature_columns", cols, now)
        self.store.set_state("training_end", f["event_time"].iloc[-1].isoformat(), now)
        self.store.set_state("training_rows", len(f), now)
        self.store.set_state("training_pending", False, now)
        self.daily_artifact.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.daily_artifact.with_suffix(".tmp")
        joblib.dump({"models":self.models,"revision":now,"training_end":f["event_time"].iloc[-1].isoformat()},temporary)
        temporary.replace(self.daily_artifact)
        self.model_revision=now
        self.store.set_state("training_revision",now,now)
        return {"rows": len(f), "features": cols, "trained_end": f["event_time"].iloc[-1].isoformat()}

    @serialized
    def train_intraday(self):
        meta = self.pqrnn_intraday.train(force=True)
        if meta.get("status") == "disabled":
            return meta
        now = datetime.now(UTC).isoformat()
        self.store.set_state("intraday_features", list(meta["features"]), now)
        self.store.set_state("intraday_training_end", self._load_intraday()["event_time"].max().isoformat(), now)
        self.store.set_state("intraday_model_type", "pqrnn", now)
        return meta

    def _event_score(self, until=None, since=None) -> float:
        until=until or datetime.now(UTC).isoformat()
        since=since or (pd.Timestamp(until)-pd.Timedelta(days=2)).isoformat()
        rows=self.store.query("SELECT severity FROM events WHERE julianday(available_time)>=julianday(?) AND julianday(available_time)<=julianday(?)",(since,until))
        return float(np.clip(sum(float(r[0]) for r in rows),0.0,5.0)/5.0) if rows else 0.0

    def _daily_forecasts(self, u, cols):
        X = u[cols].to_numpy()
        forecasts = {
            "persistence": self.models["persistence"].predict(last_return=float(u["ret1"].iloc[-1])),
            "logistic": self.models["logistic"].predict(X),
            "lightgbm_compatible": self.models["lightgbm_compatible"].predict(X),
            "arima_gjr_garch": self.models["arima_gjr_garch"].predict(),
        }
        pqrnn_fc = self.pqrnn.predict(u, cols)
        if pqrnn_fc is not None:
            forecasts["pqrnn"] = pqrnn_fc
            self.policy.register_model("pqrnn", prior=0.10)
        return forecasts

    def _persist_forecast(self, fid, created, target, horizon, last_close, out, weights, regime, conviction, payload):
        var95 = max(0.0, -out.q05)
        if self.store.query("SELECT 1 FROM forecasts WHERE forecast_id=?", (fid,)):
            return
        self.store.execute("INSERT OR IGNORE INTO forecast_identity VALUES(?,?,?,?)", (fid,payload.get("symbol", SYMBOLS["nifty"]),payload.get("reference_time",created),f"dma-{self.policy.version}"))
        row = (fid, created, target, horizon, f"dma-{self.policy.version}", str(self.policy.version), float(last_close), out.mean, out.p_up,
               out.q05, out.q25, out.q50, out.q75, out.q95, out.volatility, var95, 0.0, regime, conviction, json.dumps(payload), 0)
        self.store.execute("INSERT OR REPLACE INTO forecasts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)

    def _residual_correction(self, latest: pd.Series) -> float:
        if not self.residual.enabled or self.residual.samples < 30:
            return 0.0
        x = np.array([1.0, float(latest.get("ret1", 0.0) or 0.0), float(latest.get("ret5", 0.0) or 0.0),
                      float(latest.get("ret22", 0.0) or 0.0), float(latest.get("rv_5", 0.0) or 0.0),
                      float(latest.get("rv_22", 0.0) or 0.0)], dtype=float)
        return float(np.clip(self.residual.predict(x), -0.01, 0.01))

    @serialized
    def forecast(self):
        self.policy = DMAPolicy(self.store.get_state("policy", {}))
        self.residual = ResidualLearner(6,state=self.store.get_state("residual_learner", {}))
        if self.needs_retrain():
            self.train()
        n, e = self._load()
        f = make_daily_features(n, e)
        cols = self.store.get_state("feature_columns", [c for c in FEATURES if c in f.columns])
        u = f.dropna(subset=cols).copy()
        if len(u) < settings.min_observations:
            raise RuntimeError("Insufficient state to forecast")
        last = u.iloc[-1]
        created = datetime.now(UTC).isoformat()
        local_now = self.now_ist()
        reference_day = pd.Timestamp(last["event_time"]).date()
        last_complete = local_now.date()
        if (not self.calendar.is_market_day(local_now) or
                local_now < self.calendar.close_at(last_complete)+pd.Timedelta(minutes=settings.daily_publication_lag_minutes)):
            last_complete = self.calendar.previous_session(last_complete)
        if reference_day != last_complete:
            raise RuntimeError(f"Daily reference is stale or incomplete: expected completed session {last_complete}, received {reference_day}. Refresh history.")
        target_day = self.calendar.next_session(reference_day)
        target = self.calendar.close_at(target_day).astimezone(UTC).isoformat()
        fid = hashlib.sha1(f"{SYMBOLS['nifty']}:{last['event_time']}:1D:{target}".encode()).hexdigest()[:20]
        existing = self.store.get_state(f"forecast:{fid}")
        if existing:
            return existing
        fc = self._daily_forecasts(u, cols)
        ctx = {
            "vix": float(last.get("vix_level", 0) or 0),
            "rv22": float(last.get("rv_22", 0) or 0),
            "momentum20": float(last.get("momentum_20", 0) or 0),
            "event_severity": self._event_score(),
        }
        weights = self.policy.choose(fc, ctx)
        out = blend(fc, weights)
        correction = self._residual_correction(last)
        if correction:
            out = Forecast(out.mean + correction, out.p_up, out.q05 + correction, out.q25 + correction, out.q50 + correction, out.q75 + correction, out.q95 + correction, out.volatility, {**out.metadata, "residual_correction": correction})
        regime = "high_vol" if ctx["rv22"] > 0.015 or ctx["vix"] > 22 else ("trend" if abs(ctx["momentum20"]) > 0.004 else "calm")
        conviction = float(np.clip(1.0 - np.std([x.p_up for x in fc.values()]) * 2.0, 0.0, 1.0))
        created = datetime.now(UTC).isoformat()
        payload = {"weights": weights, "context": ctx, "experts": {k: v.__dict__ for k, v in fc.items()}, "factor_state": {k: float(last.get(k, 0) or 0) for k in cols}, "symbol": SYMBOLS["nifty"], "reference_time": last["event_time"].isoformat(), "calibration": "unvalidated live forecast"}
        self._persist_forecast(fid, created, target, "1D", last["close"], out, weights, regime, conviction, payload)
        latest = {
            "forecast_id": fid, "created_at": created, "target_time": target, "horizon": "1D", "last_close": float(last["close"]),
            "forecast": out.__dict__, "weights": weights, "regime": regime, "conviction": conviction,
            "model_disagreement": float(np.std([x.p_up for x in fc.values()])),
            "decision": self.risk.decision({"forecast": out.__dict__, "weights": weights, "conviction": conviction, "model_disagreement": float(np.std([x.p_up for x in fc.values()]))}),
        }
        latest.update({"symbol": SYMBOLS["nifty"], "reference_time": last["event_time"].isoformat(), "interval_coverage":0.90, "confidence_label":"expert agreement"})
        self.store.set_state(f"forecast:{fid}", latest, created)
        self.store.set_state("latest_forecast", latest, created)
        self.store.set_state("latest_daily_forecast", latest, created)
        self.store.set_state("policy", self.policy.state(), created)
        return latest

    @serialized
    def forecast_intraday(self):
        frame = self._load_intraday()
        if frame.empty:
            raise RuntimeError("No intraday observations")
        if not self.pqrnn_intraday.path.exists():
            raise RuntimeError("Intraday PQRNN champion is not trained; paper trading remains disabled")
        out = self.pqrnn_intraday.predict(frame)
        if out is None:
            raise RuntimeError("Intraday PQRNN could not produce a forecast")
        latest = frame.sort_values("event_time").iloc[-1]
        now = datetime.now(UTC)
        horizon_seconds = int(round(self.pqrnn_intraday._bar_seconds(frame) * self.pqrnn_intraday.cfg.horizon_bars))
        target = (pd.Timestamp(latest["event_time"]) + pd.Timedelta(seconds=60+horizon_seconds)).isoformat()
        fid = hashlib.sha1(f"{now.isoformat()}:{out.metadata['horizon']}".encode()).hexdigest()[:20]
        regime = "intraday"
        conviction = float(np.clip(2.0 * abs(out.p_up - 0.5), 0.0, 1.0))
        payload = {"weights": {"pqrnn_intraday": 1.0}, "context": {}, "experts": {"pqrnn_intraday": out.__dict__},"symbol":self.intraday_symbol,"reference_time":(pd.Timestamp(latest["event_time"])+pd.Timedelta(seconds=60)).isoformat()}
        self._persist_forecast(fid, now.isoformat(), target, out.metadata["horizon"], latest["close"], out, {"pqrnn_intraday": 1.0}, regime, conviction, payload)
        latest_state = {
            "forecast_id": fid, "created_at": now.isoformat(), "target_time": target, "horizon": out.metadata["horizon"], "last_close": float(latest["close"]),
            "forecast": out.__dict__, "weights": {"pqrnn_intraday": 1.0}, "regime": regime, "conviction": conviction,
            "decision": self.risk.decision({"forecast": out.__dict__, "weights": {"pqrnn_intraday": 1.0}, "conviction": conviction}),
        }
        self.store.set_state("latest_intraday_forecast", latest_state, now.isoformat())
        self.store.set_state("latest_forecast", latest_state if self.is_market_hours() else self.store.get_state("latest_daily_forecast", latest_state), now.isoformat())
        latest_state["p_up"] = out.p_up
        latest_state["expected_return"] = out.mean
        latest_state["volatility"] = out.volatility
        return latest_state

    @serialized
    def settle(self):
        self.policy = DMAPolicy(self.store.get_state("policy", {}))
        self.residual = ResidualLearner(6,state=self.store.get_state("residual_learner", {}))
        n, _ = self._load()
        daily = make_daily_features(n, {}).dropna(subset=["close"]) if not n.empty else pd.DataFrame(columns=["event_time","close"])
        bydate = {pd.Timestamp(r["event_time"]).date(): float(r["close"]) for _, r in daily.iterrows()}
        rows = self.store.query("SELECT * FROM forecasts WHERE settled=0 ORDER BY target_time")
        now = datetime.now(UTC).isoformat()
        settled = 0
        recent = [float(x[0]) for x in self.store.query("SELECT o.abs_error FROM outcomes o JOIN forecasts f USING(forecast_id) WHERE f.horizon='1D' ORDER BY o.settled_at DESC LIMIT 20")]
        for r in rows:
            with self.store.transaction():
                horizon = str(r["horizon"])
                due = pd.Timestamp(r["target_time"])
                lag = pd.Timedelta(minutes=settings.daily_publication_lag_minutes) if horizon == "1D" else pd.Timedelta(0)
                if pd.Timestamp(now) < due + lag:
                    continue
                identity = self.store.query("SELECT symbol FROM forecast_identity WHERE forecast_id=?", (r["forecast_id"],))
                symbol = identity[0][0] if identity else SYMBOLS["nifty"]
                actual_close = None
                if horizon == "1D":
                    target_date = pd.Timestamp(r["target_time"]).date()
                    actual_close = bydate.get(target_date)
                else:
                    rows_i = self.store.query(
                        "SELECT price FROM live_quotes WHERE symbol=? AND event_time>=? ORDER BY event_time LIMIT 1",
                        (symbol, r["target_time"]),
                    )
                    actual_close = float(rows_i[0][0]) if rows_i else None
                    if horizon == "60s":
                        target_start = (due-pd.Timedelta(seconds=60)).isoformat()
                        bar = self.store.query("SELECT close FROM instrument_bars WHERE symbol=? AND interval_seconds=60 AND event_time=?", (symbol,target_start))
                        actual_close = float(bar[0][0]) if bar else None
                    if actual_close is None and horizon == "30m":
                        # Fallback to the latest historical/intraday bar once the target time has passed.
                        latest = self.store.query("SELECT close FROM intraday_observations WHERE symbol=? AND event_time>=? ORDER BY event_time LIMIT 1", (SYMBOLS["nifty"], r["target_time"]))
                        actual_close = float(latest[0][0]) if latest else None
                if actual_close is None:
                    continue
                ret = math.log(actual_close / float(r["last_close"]))
                p = float(r["p_up"])
                brier = (p - float(ret > 0)) ** 2
                hit = int(float(r["q05"]) <= ret <= float(r["q95"]))
                ae = abs(ret - float(r["mean"]))
                direction = int((ret > 0) == (p >= 0.5))
                self.store.execute(
                    "INSERT OR REPLACE INTO outcomes VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (r["forecast_id"], ret, actual_close, now, direction, ae, ae * ae, brier, hit, abs(ret)),
                )
                self.store.execute("UPDATE forecasts SET settled=1 WHERE forecast_id=?", (r["forecast_id"],))
                try:
                    payload = json.loads(r["feature_json"])
                    experts = {k: Forecast(**v) for k, v in payload.get("experts", {}).items()}
                    shock_flag = self._event_score(until=r["target_time"],since=r["created_at"]) >= 0.45
                    if experts and horizon == "1D":
                        reason = f"settled {horizon} forecast; realized={ret:.5f}"
                        self.policy.observe(ret, experts, reason=reason)
                        self.store.execute(
                            "INSERT INTO policy_history(timestamp,policy_version,context_json,weights_json,reason) VALUES(?,?,?,?,?)",
                            (now, str(self.policy.version), json.dumps(payload.get("context", {})), json.dumps(self.policy.weights), reason),
                        )
                    shock = shock_flag
                    d = diagnose(ret, float(r["mean"]), float(r["volatility"]), float(r["q05"]), float(r["q95"]), p, list(reversed(recent)), shock)
                    self.store.execute(
                        "INSERT INTO diagnostics(forecast_id,timestamp,code,detail_json,action) VALUES(?,?,?,?,?)",
                        (r["forecast_id"], now, d.code, json.dumps({"return": ret, "error": ret - float(r["mean"]), "brier": brier, "horizon": horizon}), d.action),
                    )
                    if horizon == "1D" and d.code in {"MAGNITUDE_MISS", "DISTRIBUTION_MISS", "SYSTEMATIC_DRIFT"} and not shock_flag:
                        cols = ["ret1", "ret5", "ret22", "rv_5", "rv_22"]
                        factor = payload.get("factor_state", {})
                        if all(c in factor for c in cols):
                            x = np.array([1.0] + [float(factor[c]) for c in cols], dtype=float)
                            prior_pred = self.residual.predict(x)
                            self.residual.update(x, ret - float(r["mean"]))
                            # Enable only after the learner has had enough observations.
                            self.residual.enabled = self.residual.samples >= 30
                            self.store.set_state("residual_learner", self.residual.state(), now)
                            self.store.execute(
                                "INSERT INTO diagnostics(forecast_id,timestamp,code,detail_json,action) VALUES(?,?,?,?,?)",
                                (r["forecast_id"], now, "RESIDUAL_UPDATE", json.dumps({"samples": self.residual.samples, "prior_correction": prior_pred}), "Stored residual correction candidate"),
                            )
                except Exception as exc:
                    self.store.execute("INSERT INTO diagnostics(forecast_id,timestamp,code,detail_json,action) VALUES(?,?,?,?,?)", (r["forecast_id"], now, "DIAGNOSTIC_ERROR", json.dumps({"error": str(exc)}), "No adaptation"))
                recent.insert(0, ae)
                recent = recent[:20]
                self.store.set_state("policy",self.policy.state(),now)
                settled += 1
        self.store.set_state("policy", self.policy.state(), now)
        return settled

    def history(self, limit=120):
        prices = self.store.query("SELECT event_time,close FROM observations WHERE symbol=? ORDER BY event_time DESC LIMIT ?", (SYMBOLS["nifty"], limit))
        fc = self.store.query("SELECT forecast_id,created_at,target_time,horizon,mean,p_up,q05,q50,q95,settled FROM forecasts ORDER BY created_at DESC LIMIT ?", (limit,))
        return {
            "prices": [{"event_time": r[0], "close": float(r[1])} for r in reversed(prices)],
            "forecasts": [dict(r) for r in reversed(fc)],
        }

    def status(self):
        self.policy = DMAPolicy(self.store.get_state("policy", {}))
        def count(table):
            rows = self.store.query(f"SELECT COUNT(*) FROM {table}")
            return int(rows[0][0]) if rows else 0
        r = self.store.query("SELECT price,event_time,source FROM live_quotes WHERE symbol=? ORDER BY event_time DESC LIMIT 1", (SYMBOLS["nifty"],))
        latest = self.store.get_state("latest_forecast") or self.store.get_state("latest_daily_forecast")
        spot_time = r[0][1] if r else None
        spot_age_seconds = None
        if spot_time:
            try:
                spot_age_seconds = max(0.0, (datetime.now(UTC) - datetime.fromisoformat(spot_time.replace("Z", "+00:00"))).total_seconds())
            except ValueError:
                spot_age_seconds = None
        live_source = r[0][2] if r else (latest.get("spot_source") if latest else None)
        live_configured = self.live_provider.configured
        feed_mode = "LIVE_BROKER" if live_configured and live_source == "UpstoxV3-LTP" else "FALLBACK"
        if self.store.get_state("demo_ready"):
            feed_mode = "DEMO"
        stale_limit = 15 * 60 if self.is_market_hours() else 24 * 60 * 60
        feed_stale = spot_age_seconds is not None and spot_age_seconds > stale_limit
        feeds = {k[5:]: self.store.get_state(k) for k in ["feed:nifty_live", "feed:events", "feed:nifty", "feed:india_vix", "feed:brent", "feed:usdinr"]}
        diagnostics = self.store.query("SELECT code,action,timestamp,detail_json FROM diagnostics ORDER BY id DESC LIMIT 5")
        return {
            "observations": count("observations"), "intraday_observations": count("intraday_observations"), "live_quotes": count("live_quotes"), "events": count("events"),
            "forecasts": count("forecasts"), "outcomes": count("outcomes"), "settled": count("outcomes"),
            "policy": self.policy.state(), "latest": latest, "latest_daily": self.store.get_state("latest_daily_forecast"),
            "latest_intraday": self.store.get_state("latest_intraday_forecast"), "training_rows": self.store.get_state("training_rows", 0),
            "training_end": self.store.get_state("training_end"), "last_ingest": self.store.get_state("last_ingest"),
            "spot": float(r[0][0]) if r else (latest.get("last_close") if latest else None), "spot_time": spot_time,
            "spot_source": live_source, "live_configured": live_configured, "feed_mode": feed_mode,
            "feed_stale": feed_stale, "spot_age_seconds": spot_age_seconds,
            "live_quote": self.store.get_state("live_quote"), "risk": self.risk.decision(latest) if latest else self.store.get_state("risk_profile"),
            "risk_profile": self.risk.profile.__dict__, "feeds": feeds, "market_hours": self.is_market_hours(),
            "diagnostics": [dict(x) for x in diagnostics],
            "agent_running": bool(self.store.get_state("agent:running", False)),
            "retrain_pending": self.store.get_state("training_pending", False),
            "pqrnn": self.pqrnn.status(),
            "policy_weight_source": self.policy.state()["weight_source"],
        }

    def ledger(self, limit=200):
        rows = self.store.query("""SELECT f.*, i.symbol, i.reference_time,
            o.target_return, o.actual_close, o.direction_correct, o.abs_error, o.brier, o.interval_hit
            FROM forecasts f LEFT JOIN forecast_identity i USING(forecast_id)
            LEFT JOIN outcomes o USING(forecast_id) ORDER BY f.created_at DESC LIMIT ?""", (limit,))
        result = []
        for row in rows:
            r = dict(row)
            r.pop("feature_json", None)
            lag = pd.Timedelta(minutes=settings.daily_publication_lag_minutes) if r["horizon"] == "1D" else pd.Timedelta(0)
            due = pd.Timestamp(r["target_time"]) + lag
            r["state"] = "SETTLED" if r["settled"] else ("AWAITING_DATA" if due < pd.Timestamp.now(tz="UTC") else "PENDING")
            r["symbol"] = r["symbol"] or SYMBOLS["nifty"]
            result.append(r)
        return result

    def performance(self):
        rows = self.store.query("""SELECT f.horizon,COALESCE(i.symbol,'^NSEI') symbol,f.p_up,o.*
            FROM outcomes o JOIN forecasts f USING(forecast_id)
            LEFT JOIN forecast_identity i USING(forecast_id) ORDER BY f.target_time""")
        groups = {}
        for r in rows:
            groups.setdefault((r["symbol"],r["horizon"]), []).append(r)
        reports = []
        for (symbol,horizon), records in groups.items():
            def avg(k): return float(np.mean([r[k] for r in records]))
            tail = [r["target_return"] for r in records]
            empirical_tail = None
            if len(tail) >= 100:
                cutoff = float(np.quantile(tail,0.05))
                empirical_tail = max(0.0,-float(np.mean([x for x in tail if x<=cutoff])))
            reports.append({"symbol":symbol,"horizon":horizon,"n":len(records),"directional_accuracy":avg("direction_correct"),
                "mae":avg("abs_error"),"brier":avg("brier"),"coverage90":avg("interval_hit"),
                "log_loss":float(np.mean([-math.log(max(min(r['p_up'] if r['target_return']>0 else 1-r['p_up'],1-1e-6),1e-6)) for r in records])),
                "empirical_tail_loss95":empirical_tail,
                "interpretation":"observational; overlapping forecasts are not independent"})
        return {"groups":reports,"interval_target":0.9,"validated_profitability":False}

    def model_status(self):
        self._restore_models()
        return {"daily":{"training_rows":self.store.get_state("training_rows",0),"training_end":self.store.get_state("training_end"),
            "pending":self.store.get_state("training_pending",False),"features":self.store.get_state("feature_columns",[]),
            "loaded":bool(self.models)},"daily_neural":self.pqrnn.status(),
            "intraday":self.store.get_state("pqrnn_intraday_meta",{}),"intraday_last_run":self.store.get_state("pqrnn_intraday_last_run",{}),"intraday_champion":self.pqrnn_intraday.path.exists(),
            "intraday_rows":len(self._load_intraday()),"instrument":self.intraday_symbol}

    def _restore_models(self):
        revision=self.store.get_state("training_revision")
        if revision and revision!=self.model_revision and self.daily_artifact.exists():
            try:
                artifact=joblib.load(self.daily_artifact)
                if artifact.get("revision")==revision:
                    self.models=artifact["models"]
                    self.model_revision=revision
            except Exception as exc:
                self.store.set_state("model_restore_error",str(exc),datetime.now(UTC).isoformat())
