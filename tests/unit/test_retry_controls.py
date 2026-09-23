# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Retry controls used by load tests that must see every response unchanged:
VespaSync/feed_iterable ``num_retries_429``, VespaAsync ``docv1_retry_policy``
with ``vespa.retries.NO_RETRY``, and the HTTP status carried through errors."""

import asyncio
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
from requests.models import HTTPError

from vespa.application import Vespa, VespaAsync, VespaSync, raise_for_status
from vespa.exceptions import VespaError
from vespa.retries import NO_RETRY


def _httpr_response(status_code=200, text="{}", url="http://localhost:8080/x"):
    response = Mock()
    response.status_code = status_code
    type(response).url = PropertyMock(return_value=url)
    response.text = text
    response.json.return_value = {} if text == "{}" else {"message": text}
    return response


def test_raise_for_status_attaches_response():
    response = _httpr_response(429, text="throttled")
    with patch("vespa.application.httpr.Response", type(response)):
        with pytest.raises(VespaError) as info:
            raise_for_status(response)
    cause = info.value.__cause__
    assert isinstance(cause, HTTPError)
    assert cause.response is response
    assert cause.response.status_code == 429


@patch("vespa.application.httpr.Client")
def test_sync_num_retries_429_zero_sends_once(mock_client_class):
    client = Mock()
    mock_client_class.return_value = client
    client.get.return_value = _httpr_response(200)
    client.post.return_value = _httpr_response(429, text="throttled")

    app = Vespa(url="http://localhost", port=8080)
    with patch("vespa.application.httpr.Response", type(client.post.return_value)):
        with VespaSync(app=app, num_retries_429=0) as sync_app:
            with pytest.raises(VespaError) as info:
                sync_app.feed_data_point(schema="foo", data_id="1", fields={"a": 1})
    assert client.post.call_count == 1
    assert info.value.__cause__.response.status_code == 429


@patch("vespa.application.httpr.Client")
def test_feed_iterable_callback_sees_429_status(mock_client_class):
    client = Mock()
    mock_client_class.return_value = client
    client.get.return_value = _httpr_response(200)
    client.post.return_value = _httpr_response(429, text="throttled")
    seen = []

    app = Vespa(url="http://localhost", port=8080)
    with patch("vespa.application.httpr.Response", type(client.post.return_value)):
        app.feed_iterable(
            [{"id": "1", "fields": {"a": 1}}],
            schema="foo",
            callback=lambda response, doc_id: seen.append(response.status_code),
            num_retries_429=0,
        )
    assert seen == [429]
    assert client.post.call_count == 1


def test_async_no_retry_returns_429_after_one_attempt():
    app = Vespa(url="http://localhost", port=8080)
    session = VespaAsync(app=app, client=Mock(), docv1_retry_policy=NO_RETRY)
    session._make_request = AsyncMock(return_value=_httpr_response(429, "throttled"))

    response = asyncio.run(
        session.feed_data_point(schema="foo", data_id="1", fields={"a": 1})
    )
    assert response.status_code == 429
    assert session._make_request.await_count == 1


def test_async_default_policy_retries_429():
    app = Vespa(url="http://localhost", port=8080)
    session = VespaAsync(app=app, client=Mock())
    session._make_request = AsyncMock(
        side_effect=[_httpr_response(429, "throttled"), _httpr_response(200)]
    )
    with patch("vespa.retries.THROTTLE_RETRY.wait", return_value=0):
        response = asyncio.run(
            session.feed_data_point(schema="foo", data_id="1", fields={"a": 1})
        )
    assert response.status_code == 200
    assert session._make_request.await_count == 2
