from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 4
    base_delay: float = 0.35
    max_delay: float = 6.0
    jitter: float = 0.20
    timeout: float = 15.0


class RateLimiter:
    """Thread-safe pacing so the aggregate ATS request rate stays below provider limits."""

    def __init__(self, max_calls_per_second: float = 8.0):
        self.min_interval = 1.0 / max(1.0, max_calls_per_second)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_for = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if wait_for:
            time.sleep(wait_for)


HTTP_LIMITER = RateLimiter(8.0)
DEFAULT_RETRY = RetryConfig()


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    return isinstance(exc, (URLError, TimeoutError, OSError, ConnectionError))


def with_retries(fn: Callable[[], T], config: RetryConfig = DEFAULT_RETRY) -> T:
    last: Exception | None = None
    for attempt in range(config.attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if attempt >= config.attempts - 1 or not _retryable(exc):
                raise
            delay = min(config.max_delay, config.base_delay * (2 ** attempt))
            delay *= 1.0 + random.uniform(-config.jitter, config.jitter)
            time.sleep(max(0.05, delay))
    assert last is not None
    raise last


def fetch_json(req: Request, timeout: float | None = None, config: RetryConfig = DEFAULT_RETRY):
    timeout = timeout or config.timeout

    def _once():
        HTTP_LIMITER.wait()
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    return with_retries(_once, config)
