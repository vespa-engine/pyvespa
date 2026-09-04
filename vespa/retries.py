# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""
Retry strategies used across pyvespa.

Every retry policy in the library is defined here, once, as a tenacity
``Retrying``/``AsyncRetrying`` object. Call sites either wrap a function with
``POLICY.wraps`` (decorator style) or invoke ``POLICY.copy()(fn, ...)`` directly.
Both create a fresh copy per call, so the module-level objects are never mutated.

Defining them in one place makes the differences between strategies visible and
deliberate, gives users a documented default they can inspect or adapt via
``POLICY.copy(stop=..., wait=...)``, and gives tests a stable patch target.
"""

import random
from typing import Any, Callable

import httpr
from requests.exceptions import ConnectionError, HTTPError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_any,
    retry_if_exception,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    stop_never,
    wait_exponential,
    wait_random_exponential,
)
from tenacity.wait import wait_base

__all__ = [
    "CONTROL_PLANE_RETRY",
    "DOCV1_RETRY",
    "QUERY_RETRY",
    "SYNC_REQUEST_RETRY",
    "THROTTLE_RETRY",
    "URL_VALIDATION_RETRY",
    "VISIT_RETRY",
    "is_connection_error",
    "retry_if_status",
    "wait_golden_jitter",
]


# --------------------------------------------------------------------------- #
# Predicates and helpers
# --------------------------------------------------------------------------- #


def retry_if_status(*codes: int) -> retry_if_result:
    """
    Retry when the returned response has one of the given HTTP status codes.

    Works with any object exposing ``get_status_code()`` (``VespaResponse``) or a
    ``status_code`` attribute (``httpr.Response``).
    """

    def _has_status(response: Any) -> bool:
        get = getattr(response, "get_status_code", None)
        status = get() if callable(get) else getattr(response, "status_code", None)
        return status in codes

    return retry_if_result(_has_status)


def is_connection_error(e: BaseException) -> bool:
    """
    Check if an exception is a connection-related error.

    This handles both requests.ConnectionError and httpr exceptions
    (RequestError, ConnectError) as well as generic network errors.

    Args:
        e: The exception to check

    Returns:
        True if this is a connection/network error, False otherwise
    """
    error_str = str(e).lower()
    return (
        isinstance(e, ConnectionError)
        or isinstance(e, ConnectionResetError)
        or (hasattr(httpr, "RequestError") and isinstance(e, httpr.RequestError))
        or (hasattr(httpr, "ConnectError") and isinstance(e, httpr.ConnectError))
        or "error sending request" in error_str
        or "connection" in error_str
        or type(e).__name__ == "RequestError"
    )


def _return_last_outcome(state: RetryCallState) -> Any:
    """
    ``retry_error_callback`` that surfaces the final attempt instead of ``RetryError``.

    If the last attempt raised, that exception is re-raised. Otherwise the last
    result (e.g. a 503 response) is returned to the caller.
    """
    if state.outcome.failed:
        raise state.outcome.exception()
    return state.outcome.result()


class wait_golden_jitter(wait_base):
    """
    Wait ``0.1 * 1.618**n + uniform(0, 1)`` seconds, where ``n`` is the zero-based
    attempt index. Used by the sync client for 429 and connection-error retries.
    """

    def __call__(self, retry_state: RetryCallState) -> float:
        return 0.1 * 1.618 ** (retry_state.attempt_number - 1) + random.uniform(0, 1)


_retry_on_any_exception: Callable[[BaseException], bool] = lambda _: True  # noqa: E731

RETRY_ON_429 = retry_if_status(429)
RETRY_ON_503_OR_EXCEPTION = retry_any(
    retry_if_exception(_retry_on_any_exception), retry_if_status(503)
)


# --------------------------------------------------------------------------- #
# Policies
#
# Each constant is followed by a string literal so that both Sphinx autodoc and
# mkdocstrings (griffe) pick it up as the attribute's docstring.
# --------------------------------------------------------------------------- #

QUERY_RETRY = AsyncRetrying(
    wait=wait_random_exponential(multiplier=1.5, max=60),
    stop=stop_after_attempt(5),
)
"""Default for ``VespaAsync.query``: retry any exception up to 5 attempts with
random exponential backoff. Does not inspect the status code."""

DOCV1_RETRY = AsyncRetrying(
    wait=wait_exponential(multiplier=1),
    retry=RETRY_ON_503_OR_EXCEPTION,
    stop=stop_after_attempt(3),
    retry_error_callback=_return_last_outcome,
)
"""Outer layer for async document/v1 operations (feed/get/update/delete): retry
any exception or a 503 response up to 3 attempts. On exhaustion the last
exception is re-raised or the last response is returned, never ``RetryError``."""

THROTTLE_RETRY = AsyncRetrying(
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=RETRY_ON_429,
    stop=stop_never,
)
"""Inner layer for async document/v1 operations: retry a 429 response.

Deliberately unbounded (``stop_never``) so that sustained backpressure from
Vespa slows the feed down instead of failing it. ``feed_async_iterable`` has
no adaptive throttler, so this is the only backpressure mechanism on that path.

The wait is randomised (uniform between 0 and ``min(2**n, 10)`` seconds) so
that many concurrent tasks hitting 429 together do not retry in lockstep."""

SYNC_REQUEST_RETRY = Retrying(
    retry=retry_any(retry_if_exception(is_connection_error), RETRY_ON_429),
    wait=wait_golden_jitter(),
    stop=stop_after_attempt(11),
    retry_error_callback=_return_last_outcome,
)
"""Used by every sync data-plane request (``VespaSync._request_with_retry``):
retry a 429 response or a connection error. The stop is overridden per client
from ``VespaSync.num_retries_429`` (default 10 retries, i.e. 11 attempts).
On exhaustion the last response is returned or the last exception re-raised."""

VISIT_RETRY = Retrying(
    retry=retry_if_exception_type(HTTPError),
    stop=stop_after_attempt(3),
)
"""Per-slice retry in ``Vespa.visit`` on top of ``SYNC_REQUEST_RETRY``: retry an
``HTTPError`` raised by ``raise_for_status`` up to 3 attempts, no wait."""

CONTROL_PLANE_RETRY = Retrying(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=3),
    reraise=True,
)
"""``VespaCloud`` control-plane requests: 3 attempts, exponential wait capped at 3s."""

URL_VALIDATION_RETRY = Retrying(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
"""Validation of external model URLs in ``vespa.models``: 3 attempts, 1-10s wait."""
