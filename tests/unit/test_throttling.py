# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import asyncio
import unittest

from vespa.throttling import AdaptiveThrottler


class TestAdaptiveThrottler(unittest.TestCase):
    """Tests for the AdaptiveThrottler class."""

    def test_init_defaults(self):
        """Test default initialization."""
        throttler = AdaptiveThrottler()
        self.assertEqual(throttler.initial_concurrent, 10)
        self.assertEqual(throttler.min_concurrent, 1)
        self.assertEqual(throttler.max_concurrent, 100)
        self.assertEqual(throttler.current_concurrent, 10)
        self.assertEqual(throttler.error_threshold, 0.1)

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        throttler = AdaptiveThrottler(
            initial_concurrent=5,
            min_concurrent=2,
            max_concurrent=50,
        )
        self.assertEqual(throttler.initial_concurrent, 5)
        self.assertEqual(throttler.current_concurrent, 5)
        self.assertEqual(throttler.max_concurrent, 50)

    def test_init_caps_initial_to_max(self):
        """Test that initial_concurrent is capped at max_concurrent."""
        throttler = AdaptiveThrottler(
            initial_concurrent=100,
            max_concurrent=20,
        )
        self.assertEqual(throttler.current_concurrent, 20)

    def test_record_success_does_not_reduce(self):
        """Test that successful requests don't reduce concurrency."""

        async def run_test():
            throttler = AdaptiveThrottler(initial_concurrent=10)
            for _ in range(20):
                await throttler.record_result(200)
            return throttler.current_concurrent

        result = asyncio.run(run_test())
        self.assertEqual(result, 10)

    def test_record_errors_reduces_concurrency(self):
        """Test that high error rate reduces concurrency."""

        async def run_test():
            throttler = AdaptiveThrottler(
                initial_concurrent=10,
                error_threshold=0.1,
                reduction_factor=0.5,
            )
            # Mix of successes and errors (>10% errors)
            for _ in range(8):
                await throttler.record_result(200)
            for _ in range(3):
                await throttler.record_result(504)
            return throttler.current_concurrent

        result = asyncio.run(run_test())
        # Should have reduced: 10 * 0.5 = 5
        self.assertEqual(result, 5)

    def test_reduces_on_503(self):
        """Test that 503 Service Unavailable triggers reduction."""

        async def run_test():
            throttler = AdaptiveThrottler(initial_concurrent=10)
            for _ in range(5):
                await throttler.record_result(200)
            for _ in range(6):
                await throttler.record_result(503)
            return throttler.current_concurrent

        result = asyncio.run(run_test())
        self.assertLess(result, 10)

    def test_reduces_on_429(self):
        """Test that 429 Too Many Requests triggers reduction."""

        async def run_test():
            throttler = AdaptiveThrottler(initial_concurrent=10)
            for _ in range(5):
                await throttler.record_result(200)
            for _ in range(6):
                await throttler.record_result(429)
            return throttler.current_concurrent

        result = asyncio.run(run_test())
        self.assertLess(result, 10)

    def test_minimum_concurrency_floor(self):
        """Test that concurrency doesn't go below min_concurrent."""

        async def run_test():
            throttler = AdaptiveThrottler(
                initial_concurrent=4,
                min_concurrent=2,
                reduction_factor=0.5,
            )
            # Force multiple reductions
            for _ in range(3):
                for _ in range(5):
                    await throttler.record_result(200)
                for _ in range(6):
                    await throttler.record_result(504)
            return throttler.current_concurrent

        result = asyncio.run(run_test())
        self.assertGreaterEqual(result, 2)

    def test_increase_after_cooldown(self):
        """Test that concurrency increases after successful period."""

        async def run_test():
            throttler = AdaptiveThrottler(
                initial_concurrent=5,
                max_concurrent=20,
                success_window=10,
                increase_step=2,
                cooldown_seconds=0.1,
            )
            # Force a reduction first
            for _ in range(5):
                await throttler.record_result(200)
            for _ in range(6):
                await throttler.record_result(504)

            reduced = throttler.current_concurrent

            # Wait for cooldown
            await asyncio.sleep(0.15)

            # Record enough successes to trigger increase
            for _ in range(15):
                await throttler.record_result(200)

            return reduced, throttler.current_concurrent

        reduced, final = asyncio.run(run_test())
        self.assertGreaterEqual(final, reduced)

    def test_semaphore_property(self):
        """Test that semaphore property returns valid semaphore."""
        throttler = AdaptiveThrottler(initial_concurrent=5)
        self.assertIsInstance(throttler.semaphore, asyncio.Semaphore)

    def test_reset(self):
        """Test that reset returns throttler to initial state."""
        throttler = AdaptiveThrottler(initial_concurrent=10, max_concurrent=50)

        # Manually modify state
        throttler._current_concurrent = 5
        throttler._window = [True, False, True]
        throttler._last_reduction = 12345.0

        throttler.reset()

        self.assertEqual(throttler.current_concurrent, 10)
        self.assertEqual(len(throttler._window), 0)
        self.assertEqual(throttler._last_reduction, 0.0)

    def test_window_bounded(self):
        """Test that result window is bounded to 100 entries."""

        async def run_test():
            throttler = AdaptiveThrottler(initial_concurrent=10)
            for _ in range(150):
                await throttler.record_result(200)
            return len(throttler._window)

        result = asyncio.run(run_test())
        self.assertLessEqual(result, 100)

    def test_is_error_status(self):
        """Test error status detection."""
        throttler = AdaptiveThrottler()

        # Error statuses
        self.assertTrue(throttler._is_error_status(429))
        self.assertTrue(throttler._is_error_status(503))
        self.assertTrue(throttler._is_error_status(504))
        self.assertTrue(throttler._is_error_status(500))
        self.assertTrue(throttler._is_error_status(502))

        # Success statuses
        self.assertFalse(throttler._is_error_status(200))
        self.assertFalse(throttler._is_error_status(201))
        self.assertFalse(throttler._is_error_status(400))
        self.assertFalse(throttler._is_error_status(404))


if __name__ == "__main__":
    unittest.main()
