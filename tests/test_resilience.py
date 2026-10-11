import time
from urllib.error import HTTPError
from src.ats.resilience import RateLimiter, RetryConfig, with_retries


def test_retries_transient_error():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("temporary")
        return 42
    assert with_retries(fn, RetryConfig(attempts=3, base_delay=0.01, max_delay=0.02, jitter=0)) == 42
    assert len(calls) == 3


def test_rate_limiter_paces_calls():
    limiter = RateLimiter(20)
    t0 = time.monotonic()
    limiter.wait(); limiter.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.045
