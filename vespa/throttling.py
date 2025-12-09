# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""
Adaptive throttling for async requests to Vespa.

This module provides an AdaptiveThrottler class that dynamically adjusts
concurrency based on server response patterns to prevent overloading
Vespa applications with expensive operations (e.g., large embedding models).
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class AdaptiveThrottler:
    """
    Adaptive throttler that adjusts concurrency based on response status codes.

    The throttler starts with a conservative concurrency limit and automatically
    adjusts based on server responses:
    - Reduces concurrency when error rate exceeds threshold (504, 503, 429 errors)
    - Gradually increases concurrency during healthy periods after a cooldown

    Attributes:
        initial_concurrent: Starting concurrency limit (default: 10)
        min_concurrent: Minimum concurrency floor (default: 1)
        max_concurrent: Maximum concurrency ceiling (default: 100)
        error_threshold: Error rate that triggers reduction (default: 0.1 = 10%)
        success_window: Consecutive successes needed to increase (default: 50)
        reduction_factor: Factor to reduce concurrency by (default: 0.5 = 50%)
        increase_step: Amount to increase on success (default: 2)
        cooldown_seconds: Wait time before increasing after reduction (default: 5.0)

    Example:
        ```python
        throttler = AdaptiveThrottler(initial_concurrent=10, max_concurrent=50)

        async def make_request():
            async with throttler.semaphore:
                response = await do_request()
                await throttler.record_result(response.status_code)
                return response
        ```
    """

    initial_concurrent: int = 10
    min_concurrent: int = 1
    max_concurrent: int = 100
    error_threshold: float = 0.1
    success_window: int = 50
    reduction_factor: float = 0.5
    increase_step: int = 2
    cooldown_seconds: float = 5.0

    # Internal state (not part of __init__ signature)
    _current_concurrent: int = field(init=False, repr=False)
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _window: List[bool] = field(init=False, repr=False, default_factory=list)
    _last_reduction: float = field(init=False, repr=False, default=0.0)
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize internal state after dataclass construction."""
        self._current_concurrent = min(self.initial_concurrent, self.max_concurrent)
        self._semaphore = asyncio.Semaphore(self._current_concurrent)
        self._window = []
        self._last_reduction = 0.0
        self._lock = asyncio.Lock()

    @property
    def current_concurrent(self) -> int:
        """Current concurrency limit."""
        return self._current_concurrent

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Async semaphore for rate limiting requests."""
        return self._semaphore

    def _is_error_status(self, status_code: int) -> bool:
        """Check if status code indicates server overload."""
        return status_code in (429, 503, 504) or status_code >= 500

    async def record_result(self, status_code: int) -> None:
        """
        Record a request result and adjust throttling if needed.

        Args:
            status_code: HTTP status code from the response
        """
        async with self._lock:
            is_success = not self._is_error_status(status_code)
            self._window.append(is_success)

            # Keep window bounded to last 100 results
            if len(self._window) > 100:
                self._window = self._window[-100:]

            # Check error rate after minimum sample size
            if len(self._window) >= 10:
                error_count = self._window.count(False)
                error_rate = error_count / len(self._window)

                if error_rate > self.error_threshold:
                    await self._reduce_concurrency()
                elif self._should_increase():
                    await self._increase_concurrency()

    def _should_increase(self) -> bool:
        """Check if conditions are met to increase concurrency."""
        if len(self._window) < self.success_window:
            return False

        # Check last N results are all successes
        recent = self._window[-self.success_window :]
        if all(recent):
            # Check cooldown period has elapsed
            return time.time() - self._last_reduction >= self.cooldown_seconds
        return False

    async def _reduce_concurrency(self) -> None:
        """Reduce concurrency when errors are detected."""
        new_limit = max(
            self.min_concurrent, int(self._current_concurrent * self.reduction_factor)
        )
        if new_limit < self._current_concurrent:
            self._current_concurrent = new_limit
            self._semaphore = asyncio.Semaphore(new_limit)
            self._last_reduction = time.time()
            self._window.clear()  # Reset window after adjustment

    async def _increase_concurrency(self) -> None:
        """Gradually increase concurrency during healthy periods."""
        new_limit = min(self.max_concurrent, self._current_concurrent + self.increase_step)
        if new_limit > self._current_concurrent:
            self._current_concurrent = new_limit
            self._semaphore = asyncio.Semaphore(new_limit)
            self._window.clear()  # Reset window after adjustment

    def reset(self) -> None:
        """Reset throttler to initial state."""
        self._current_concurrent = min(self.initial_concurrent, self.max_concurrent)
        self._semaphore = asyncio.Semaphore(self._current_concurrent)
        self._window.clear()
        self._last_reduction = 0.0
