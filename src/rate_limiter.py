import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Tuple


class TokenBucketRateLimiter:
    """Thread-safe sliding-window RPM/TPM + daily RPD rate limiter."""

    def __init__(self, max_rpm: int, max_tpm: int, max_rpd: int) -> None:
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.max_rpd = max_rpd
        self._lock = threading.Lock()
        self._events: deque[Tuple[float, int]] = deque()
        self._day_key = self._current_day()
        self._rpd_count = 0

    @staticmethod
    def _current_day() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _prune_old(self, now: float) -> None:
        cutoff = now - 60.0
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        today = self._current_day()
        if today != self._day_key:
            self._day_key = today
            self._rpd_count = 0

    def acquire(self, estimated_tokens: int = 1500) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune_old(now)

                if self._rpd_count >= self.max_rpd:
                    raise RuntimeError(
                        f"Daily free-tier LLM cap reached ({self._rpd_count}/{self.max_rpd} RPD)."
                    )

                rpm_used = len(self._events)
                tpm_used = sum(tok for _, tok in self._events)

                if rpm_used < self.max_rpm and (tpm_used + estimated_tokens) <= self.max_tpm:
                    self._events.append((now, estimated_tokens))
                    self._rpd_count += 1
                    return

                oldest_ts = self._events[0][0] if self._events else now
                sleep_for = max(0.5, (oldest_ts + 60.0) - now + random.uniform(0.1, 0.6))

            time.sleep(sleep_for)

    def status(self) -> Dict[str, int]:
        with self._lock:
            self._prune_old(time.monotonic())
            return {
                "rpm_used": len(self._events),
                "tpm_used": sum(tok for _, tok in self._events),
                "rpd_used": self._rpd_count,
            }


def _extract_retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers and "Retry-After" in headers:
        try:
            return float(headers["Retry-After"])
        except ValueError:
            return None
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg


def execute_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 4,
    base_delay: float = 4.0,
    max_delay: float = 60.0,
    **kwargs: Any,
) -> Any:
    """Executes an API call with HTTP 429 Retry-After respect + jittered exponential backoff."""
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not _is_rate_limit_error(exc):
                raise

            retry_after = _extract_retry_after(exc)
            if retry_after is not None:
                wait_seconds = min(max_delay, retry_after + random.uniform(0.25, 1.0))
            else:
                exp_wait = min(max_delay, base_delay * (2 ** (attempt - 1)))
                wait_seconds = exp_wait + random.uniform(0.25, 1.5)

            time.sleep(wait_seconds)
