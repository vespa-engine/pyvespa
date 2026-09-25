## `vespa.throttling`

Adaptive throttling for async requests to Vespa.

This module provides an AdaptiveThrottler class that dynamically adjusts concurrency based on server response patterns to prevent overloading Vespa applications with expensive operations (e.g., large embedding models).

### `AdaptiveThrottler(initial_concurrent=10, min_concurrent=1, max_concurrent=100, error_threshold=0.1, success_window=50, reduction_factor=0.5, increase_step=2, cooldown_seconds=5.0)`

Adaptive throttler that adjusts concurrency based on response status codes.

The throttler starts with a conservative concurrency limit and automatically adjusts based on server responses:

- Reduces concurrency when error rate exceeds threshold (504, 503, 429 errors)
- Gradually increases concurrency during healthy periods after a cooldown

Attributes:

| Name                 | Type    | Description                                                |
| -------------------- | ------- | ---------------------------------------------------------- |
| `initial_concurrent` | `int`   | Starting concurrency limit (default: 10)                   |
| `min_concurrent`     | `int`   | Minimum concurrency floor (default: 1)                     |
| `max_concurrent`     | `int`   | Maximum concurrency ceiling (default: 100)                 |
| `error_threshold`    | `float` | Error rate that triggers reduction (default: 0.1 = 10%)    |
| `success_window`     | `int`   | Consecutive successes needed to increase (default: 50)     |
| `reduction_factor`   | `float` | Factor to reduce concurrency by (default: 0.5 = 50%)       |
| `increase_step`      | `int`   | Amount to increase on success (default: 2)                 |
| `cooldown_seconds`   | `float` | Wait time before increasing after reduction (default: 5.0) |

Example

```python
throttler = AdaptiveThrottler(initial_concurrent=10, max_concurrent=50)

async def make_request():
    async with throttler.semaphore:
        response = await do_request()
        await throttler.record_result(response.status_code)
        return response
```

#### `current_concurrent`

Current concurrency limit.

#### `semaphore`

Async semaphore for rate limiting requests.

Note: This property lazily creates the semaphore on first access to maintain compatibility with Python 3.9.

#### `__post_init__()`

Initialize internal state after dataclass construction.

#### `record_result(status_code)`

Record a request result and adjust throttling if needed.

Parameters:

| Name          | Type  | Description                        | Default    |
| ------------- | ----- | ---------------------------------- | ---------- |
| `status_code` | `int` | HTTP status code from the response | *required* |

#### `reset()`

Reset throttler to initial state.
