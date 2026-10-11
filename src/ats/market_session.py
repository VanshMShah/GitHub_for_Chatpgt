from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NSE 2026 capital-market trading holidays. Source: NSE Market Timings & Holidays,
# including the exchange-notified 2026 holiday calendar. Keep this explicit and
# versioned so a future calendar change cannot silently alter experiment timing.
NSE_2026_HOLIDAYS = frozenset({
    date(2026, 1, 15),  # Maharashtra municipal corporation election
    date(2026, 1, 26),
    date(2026, 2, 19),
    date(2026, 3, 3),
    date(2026, 3, 19),
    date(2026, 3, 26),
    date(2026, 3, 31),
    date(2026, 4, 1),
    date(2026, 4, 3),
    date(2026, 4, 14),
    date(2026, 5, 1),
    date(2026, 5, 28),
    date(2026, 6, 26),
    date(2026, 8, 26),
    date(2026, 9, 14),
    date(2026, 10, 2),
    date(2026, 10, 20),
    date(2026, 11, 10),
    date(2026, 11, 24),
    date(2026, 12, 25),
})


@dataclass(frozen=True)
class MarketSession:
    open_time: time = time(9, 15)
    close_time: time = time(15, 30)
    holidays: frozenset[date] = NSE_2026_HOLIDAYS

    def is_market_day(self, now: datetime) -> bool:
        local = now.astimezone(IST)
        if local.year != 2026:
            raise RuntimeError(f"NSE holiday calendar not provisioned for {local.year}; fail closed")
        return local.weekday() < 5 and local.date() not in self.holidays

    def is_open(self, now: datetime) -> bool:
        local = now.astimezone(IST)
        return self.is_market_day(local) and self.open_time <= local.time() <= self.close_time

    def session_date(self, now: datetime) -> str | None:
        local = now.astimezone(IST)
        return local.date().isoformat() if self.is_market_day(local) else None

    def next_session(self, day: date) -> date:
        from datetime import timedelta
        candidate = day
        for _ in range(370):
            candidate += timedelta(days=1)
            if self.is_market_day(datetime.combine(candidate, self.open_time, tzinfo=IST)):
                return candidate
        raise RuntimeError("No supported next session")

    def close_at(self, day: date) -> datetime:
        return datetime.combine(day, self.close_time, tzinfo=IST)

    def previous_session(self, day: date) -> date:
        from datetime import timedelta
        candidate = day
        for _ in range(370):
            candidate -= timedelta(days=1)
            if self.is_market_day(datetime.combine(candidate, self.open_time, tzinfo=IST)):
                return candidate
        raise RuntimeError("No supported previous session")
