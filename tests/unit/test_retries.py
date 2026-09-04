# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""
Tests for vespa.retries and the call sites that use it.

Waits are neutralised by patching the module-level policy constants at their
point of use with ``POLICY.copy(wait=wait_none())``. Because every call site
copies the policy before invoking it, this is safe and needs no tenacity internals.
"""

from unittest.mock import AsyncMock, Mock, patch

import httpr
import pytest
from requests.exceptions import HTTPError
from tenacity import (
    AsyncRetrying,
    Retrying,
    stop_after_attempt,
    stop_never,
    wait_none,
    wait_random_exponential,
)
from tenacity.stop import stop_after_attempt as _stop_after_attempt_cls

from vespa import retries
from vespa.application import Vespa, VespaAsync, VespaSync
from vespa.io import VespaResponse
from vespa.retries import (
    CONTROL_PLANE_RETRY,
    DOCV1_RETRY,
    QUERY_RETRY,
    SYNC_REQUEST_RETRY,
    THROTTLE_RETRY,
    URL_VALIDATION_RETRY,
    VISIT_RETRY,
    is_connection_error,
    retry_if_status,
    wait_golden_jitter,
)


def _httpr_response(status_code: int, json_data=None) -> Mock:
    response = Mock(spec=httpr.Response)
    response.status_code = status_code
    response.json.return_value = json_data if json_data is not None else {}
    response.text = "{}"
    response.url = "http://localhost:8080/"
    response.headers = {}
    return response


class TestPolicyDefinitions:
    """The documented shape of each policy, so drift is caught by a test."""

    def test_async_policies_are_async(self):
        for policy in (QUERY_RETRY, DOCV1_RETRY, THROTTLE_RETRY):
            assert isinstance(policy, AsyncRetrying)

    def test_sync_policies_are_sync(self):
        for policy in (
            SYNC_REQUEST_RETRY,
            VISIT_RETRY,
            CONTROL_PLANE_RETRY,
            URL_VALIDATION_RETRY,
        ):
            assert isinstance(policy, Retrying)

    def test_bounded_policies_stop_after_attempt(self):
        expected = {
            QUERY_RETRY: 5,
            DOCV1_RETRY: 3,
            SYNC_REQUEST_RETRY: 11,
            VISIT_RETRY: 3,
            CONTROL_PLANE_RETRY: 3,
            URL_VALIDATION_RETRY: 3,
        }
        for policy, attempts in expected.items():
            assert isinstance(policy.stop, _stop_after_attempt_cls)
            assert policy.stop.max_attempt_number == attempts

    def test_throttle_retry_is_deliberately_unbounded(self):
        # Explicit decision, see the THROTTLE_RETRY docstring in vespa.retries.
        assert THROTTLE_RETRY.stop is stop_never

    def test_throttle_retry_uses_jittered_backoff(self):
        # All four docv1 methods share THROTTLE_RETRY. The wait must stay
        # randomised so concurrent tasks that hit 429 together do not retry in
        # lockstep (see review on #1346); assert type and cap so a silent
        # change is noticed.
        assert isinstance(THROTTLE_RETRY.wait, wait_random_exponential)
        assert THROTTLE_RETRY.wait.multiplier == 1
        assert THROTTLE_RETRY.wait.max == 10

    def test_public_constants_are_exported(self):
        for name in retries.__all__:
            assert hasattr(retries, name)


class TestHelpers:
    def test_retry_if_status_on_vespa_response(self):
        predicate = retry_if_status(429, 503).predicate
        assert predicate(
            VespaResponse(json={}, status_code=429, url="", operation_type="get")
        )
        assert predicate(
            VespaResponse(json={}, status_code=503, url="", operation_type="get")
        )
        assert not predicate(
            VespaResponse(json={}, status_code=200, url="", operation_type="get")
        )

    def test_retry_if_status_on_httpr_response(self):
        predicate = retry_if_status(429).predicate
        assert predicate(_httpr_response(429))
        assert not predicate(_httpr_response(200))

    def test_wait_golden_jitter_matches_legacy_formula(self):
        state = Mock()
        with patch("vespa.retries.random.uniform", return_value=0.0):
            state.attempt_number = 1
            assert wait_golden_jitter()(state) == pytest.approx(0.1)
            state.attempt_number = 4
            assert wait_golden_jitter()(state) == pytest.approx(0.1 * 1.618**3)

    def test_is_connection_error(self):
        assert is_connection_error(ConnectionResetError())
        assert is_connection_error(RuntimeError("error sending request"))
        assert not is_connection_error(ValueError("bad value"))


class TestSyncRequestRetry:
    """VespaSync._request_with_retry keeps its pre-tenacity behaviour."""

    @pytest.fixture
    def sync_client(self):
        app = Vespa(url="http://localhost", port=8080)
        client = VespaSync(app)
        client.http_client = Mock()
        return client

    @pytest.fixture(autouse=True)
    def no_wait(self):
        with patch(
            "vespa.application.SYNC_REQUEST_RETRY",
            SYNC_REQUEST_RETRY.copy(wait=wait_none()),
        ):
            yield

    def test_retries_429_then_returns_success(self, sync_client):
        sync_client.http_client.get.side_effect = [
            _httpr_response(429),
            _httpr_response(429),
            _httpr_response(200),
        ]
        response = sync_client._request_with_retry("GET", "http://x")
        assert response.status_code == 200
        assert sync_client.http_client.get.call_count == 3

    def test_returns_last_429_after_num_retries(self, sync_client):
        sync_client.num_retries_429 = 2
        sync_client.http_client.get.return_value = _httpr_response(429)
        response = sync_client._request_with_retry("GET", "http://x")
        assert response.status_code == 429
        assert sync_client.http_client.get.call_count == 3  # 1 + num_retries_429

    def test_retries_connection_error_then_reraises(self, sync_client):
        sync_client.num_retries_429 = 1
        sync_client.http_client.post.side_effect = ConnectionResetError("reset")
        with pytest.raises(ConnectionResetError):
            sync_client._request_with_retry("POST", "http://x", json_data={"a": 1})
        assert sync_client.http_client.post.call_count == 2

    def test_non_connection_error_raises_immediately(self, sync_client):
        sync_client.http_client.get.side_effect = ValueError("boom")
        with pytest.raises(ValueError):
            sync_client._request_with_retry("GET", "http://x")
        assert sync_client.http_client.get.call_count == 1

    def test_does_not_retry_5xx(self, sync_client):
        sync_client.http_client.get.return_value = _httpr_response(503)
        response = sync_client._request_with_retry("GET", "http://x")
        assert response.status_code == 503
        assert sync_client.http_client.get.call_count == 1


class TestVisitRetry:
    def test_visit_policy_retries_http_error_three_times(self):
        calls = {"n": 0}

        @VISIT_RETRY.wraps
        def flaky():
            calls["n"] += 1
            raise HTTPError("HTTP 500")

        with pytest.raises(Exception):
            flaky()
        assert calls["n"] == 3


@pytest.mark.asyncio
class TestAsyncDocv1Retry:
    """The two-layer docv1 policy behaves like the former stacked decorators."""

    @pytest.fixture
    def async_client(self):
        app = Vespa(url="http://localhost", port=8080)
        return VespaAsync(app)

    @pytest.fixture(autouse=True)
    def no_wait(self):
        with (
            patch("vespa.application.DOCV1_RETRY", DOCV1_RETRY.copy(wait=wait_none())),
            patch(
                "vespa.application.THROTTLE_RETRY",
                THROTTLE_RETRY.copy(wait=wait_none()),
            ),
        ):
            yield

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("feed_data_point", {"fields": {"a": 1}}),
            ("update_data", {"fields": {"a": 1}}),
            ("get_data", {}),
            ("delete_data", {}),
        ],
    )
    async def test_retries_429_until_success(self, async_client, method, kwargs):
        async_client._make_request = AsyncMock(
            side_effect=[_httpr_response(429)] * 4 + [_httpr_response(200)]
        )
        response = await getattr(async_client, method)(
            schema="s", data_id="1", **kwargs
        )
        assert response.status_code == 200
        assert async_client._make_request.await_count == 5

    async def test_429_retry_is_unbounded(self, async_client):
        # Well past any small stop bound: 429 must keep retrying until success.
        async_client._make_request = AsyncMock(
            side_effect=[_httpr_response(429)] * 25 + [_httpr_response(200)]
        )
        response = await async_client.get_data(schema="s", data_id="1")
        assert response.status_code == 200
        assert async_client._make_request.await_count == 26

    async def test_503_retried_three_times_then_returned(self, async_client):
        async_client._make_request = AsyncMock(return_value=_httpr_response(503))
        response = await async_client.feed_data_point(
            schema="s", data_id="1", fields={"a": 1}
        )
        assert response.status_code == 503
        assert async_client._make_request.await_count == 3

    async def test_exception_retried_three_times_then_reraised(self, async_client):
        async_client._make_request = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            await async_client.delete_data(schema="s", data_id="1")
        assert async_client._make_request.await_count == 3

    async def test_operation_type_and_payload_preserved(self, async_client):
        async_client._make_request = AsyncMock(return_value=_httpr_response(200))
        response = await async_client.update_data(
            schema="s", data_id="1", fields={"a": 1}, create=True
        )
        assert response.operation_type == "update"
        call = async_client._make_request.await_args
        assert call.args[0] == "PUT"
        assert call.kwargs["json_data"] == {"fields": {"a": {"assign": 1}}}
        assert call.kwargs["params"] == {"create": "true"}

    async def test_custom_stop_via_copy(self):
        # Users can bound the 429 retry themselves by copying the constant.
        bounded = THROTTLE_RETRY.copy(stop=stop_after_attempt(2), wait=wait_none())
        assert bounded.stop.max_attempt_number == 2
        assert THROTTLE_RETRY.stop is stop_never  # original untouched
