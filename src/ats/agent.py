from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone

from src.ats.config import settings

log = logging.getLogger("ats.agent")


class BackgroundAgent:
    """Persistent worker with idempotent retries and safe degradation."""

    def __init__(self, engine):
        self.engine = engine
        self.stop_event = asyncio.Event()
        self.ready = False
        self.last_forecast_epoch = 0.0
        self._failure_streak = 0
        self._retry_lock = asyncio.Lock()

    async def _run_component(self, name, fn, attempts: int = 3):
        last = None
        for attempt in range(1, attempts + 1):
            try:
                work = asyncio.create_task(asyncio.to_thread(fn))
                try:
                    result = await asyncio.shield(work)
                except asyncio.CancelledError:
                    await work
                    raise
                self.engine.store.set_state(
                    f"agent:{name}",
                    {"ok": True, "attempt": attempt, "updated_at": datetime.now(timezone.utc).isoformat()},
                    datetime.now(timezone.utc).isoformat(),
                )
                return result
            except Exception as exc:
                last = exc
                self.engine.store.set_state(
                    f"agent:{name}",
                    {"ok": False, "attempt": attempt, "error": str(exc), "updated_at": datetime.now(timezone.utc).isoformat()},
                    datetime.now(timezone.utc).isoformat(),
                )
                if attempt < attempts:
                    delay = min(20.0, 0.8 * (2 ** (attempt - 1))) * (1.0 + random.uniform(-0.15, 0.15))
                    log.warning("%s failed attempt %s/%s: %s; retrying in %.1fs", name, attempt, attempts, exc, delay)
                    await asyncio.sleep(max(0.2, delay))
        raise last

    async def tick(self):
        async with self._retry_lock:
            try:
                now = datetime.now(timezone.utc).timestamp()
                # A complete cycle is safe to repeat because writes are keyed by event/forecast ids.
                result = await self._run_component("sync", lambda: self.engine.sync(False), attempts=3)
                if result.get("settled"):
                    log.info("settled=%s", result["settled"])
                self.ready = bool(self.engine.models)

                forecast_interval = settings.forecast_interval_market_seconds if self.engine.is_market_hours() else settings.forecast_interval_offmarket_seconds
                if now - self.last_forecast_epoch >= forecast_interval:
                    await self._run_component("forecast",self.engine.forecast,attempts=3)
                    self.last_forecast_epoch=now
                self.engine.store.set_state("agent:last_success",datetime.now(timezone.utc).isoformat(),datetime.now(timezone.utc).isoformat())
                self._failure_streak = 0
            except Exception as exc:
                self._failure_streak += 1
                log.exception("ATS cycle failed streak=%s: %s", self._failure_streak, exc)
                self.engine.store.set_state(
                    "agent:last_failure",
                    {"streak": self._failure_streak, "error": str(exc), "updated_at": datetime.now(timezone.utc).isoformat()},
                    datetime.now(timezone.utc).isoformat(),
                )

    async def run(self):
        await self.tick()
        while not self.stop_event.is_set():
            interval = min(settings.live_poll_seconds, 30) if self.engine.is_market_hours() else min(settings.offmarket_poll_seconds, 300)
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                await self.tick()

    def stop(self):
        self.stop_event.set()
