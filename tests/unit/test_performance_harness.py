# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Unit tests for the performance-lane harness under tests/performance/utils:
session concurrency derivation, worker aggregation, k6 summary parsing and the
instance CPU probe parser. No network, no k6."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "performance"))

from utils.k6_lane import lane_result  # noqa: E402
from utils.loadgen import WorkerResult, aggregate  # noqa: E402
from utils.metrics import (  # noqa: E402
    LaneResult,
    ValidityLimits,
    assert_measurement_valid,
)
from utils.saturation import _CPU_UTIL  # noqa: E402
from utils.workloads import LoadProfile  # noqa: E402


def test_for_session_puts_the_queue_target_inside_the_instance():
    profile = LoadProfile(processes=8, server_queue_target=250)
    # N_total = target + ceiling * rtt, split over two transports, whole processes.
    us = profile.for_session(ceiling_rps=4058, rtt_s=0.056)
    europe = profile.for_session(ceiling_rps=4400, rtt_s=0.125)
    assert us.concurrency == 240
    assert europe.concurrency == 400
    assert us.connections() == 8 and us.streams_per_connection() == 30
    # Unmeasurable inputs keep the default.
    assert profile.for_session(0, 0).concurrency == profile.concurrency
    assert profile.for_session(10**6, 10).concurrency == profile.max_concurrency


def test_per_process_splits_concurrency_and_batch():
    share = LoadProfile(
        concurrency=400, processes=8, iterable_docs=300000
    ).per_process()
    assert share.concurrency == 50
    assert share.processes == 1
    assert share.iterable_docs == 37500
    assert (
        LoadProfile(concurrency=400, processes=8).k6_env()["STREAMS_PER_CONNECTION"]
        == "50"
    )


def _worker(requests, errors, latencies, started, finished, cpu_s, completions=()):
    return WorkerResult(
        requests=requests,
        errors=errors,
        rate_limited=0,
        latencies_ms=latencies,
        started=started,
        finished=finished,
        cpu_s=cpu_s,
        completions=list(completions),
        status_counts={"200": requests - errors, "500": errors}
        if errors
        else {"200": requests},
    )


def test_aggregate_closed_loop_uses_the_window_and_sums_workers():
    profile = LoadProfile(concurrency=100, processes=2, duration_s=10.0)
    parts = [
        _worker(600, 6, [10.0] * 600, 0.0, 12.0, 1.0),
        _worker(400, 0, [30.0] * 400, 0.0, 12.0, 1.0),
    ]
    result = aggregate("sync_feed_data_point", "token", parts, profile)
    assert result.requests == 1000
    assert result.rps == pytest.approx(100.0)  # 1000 / duration_s
    assert result.error_rate == pytest.approx(0.006)
    assert result.p50_ms == 10.0 and result.p99_ms == 30.0
    assert result.cpu_ms_per_request == pytest.approx(2.0)  # 2 s over 1000
    assert result.connections == 2 and result.processes == 2
    assert result.status_counts == {"200": 994, "500": 6}


def test_aggregate_batch_counts_only_while_every_process_feeds():
    profile = LoadProfile(concurrency=100, processes=2)
    # Process A feeds 0..10 s, process B 2..8 s: window is [2, 8].
    a = [(t, 200) for t in (1.0, 3.0, 5.0, 7.0, 9.0)]
    b = [(t, 200) for t in (2.5, 4.5, 6.5)] + [(7.5, 429)]
    parts = [
        _worker(5, 0, [], 0.0, 10.0, 0.5, a),
        _worker(4, 1, [], 2.0, 8.0, 0.5, b),
    ]
    result = aggregate("feed_iterable", "mtls", parts, profile)
    assert result.duration_s == pytest.approx(6.0)
    assert result.requests == 7  # 3 from A inside [2, 8], 4 from B
    assert result.rps == pytest.approx(7 / 6)
    assert result.rate_limited_rate == pytest.approx(1 / 7)
    assert result.error_rate == pytest.approx(1 / 7)
    assert result.p50_ms is None


def test_k6_lane_result_uses_the_hold_window_not_the_summary_rate():
    profile = LoadProfile(concurrency=240, processes=8, duration_s=150.0)
    metrics = {
        "token_req_duration": {"med": 100.0, "p(95)": 160.0, "p(99)": 200.0},
        "token_fail_rate": {"value": 0.005},
        "token_reqs": {"count": 300000, "rate": 1500.0},  # rate is over the whole run
        "token_rate_limited": {"count": 1500},
    }
    result = lane_result(metrics, "token", profile)
    assert result.rps == pytest.approx(2000.0)
    assert result.rate_limited_rate == pytest.approx(0.005)
    assert result.connections == 8
    # A 429 counter k6 never emitted means zero.
    del metrics["token_rate_limited"]
    assert lane_result(metrics, "token", profile).rate_limited_rate == 0.0


def test_cpu_util_regex_reads_value_and_timestamp():
    line = (
        'cpu_util{applicationId="t.a.d",clusterId="container/msmarco_container",'
        'vespa_service="vespa_node",} 87.5 1790061567000\n'
        'cpu_util{clusterId="content/msmarco_content",} 42.0\n'
    )
    found = _CPU_UTIL.findall(line)
    assert found[0][1] == "87.5" and found[0][2] == "1790061567000"
    assert found[1][1] == "42.0" and found[1][2] == ""


def _result(**overrides):
    base = dict(
        lane="pyvespa",
        method="m",
        transport="token",
        http="h2only",
        rps=1000.0,
        error_rate=0.0,
        requests=1,
        duration_s=1.0,
        concurrency=1,
    )
    base.update(overrides)
    return LaneResult(**base)


def test_validity_names_the_side_that_was_the_bottleneck():
    limits = ValidityLimits(
        max_rate_limited_rate=0.01,
        max_client_cpu_fraction=0.7,
        min_server_container_cpu_util=0.75,
    )
    assert_measurement_valid(
        [
            _result(
                rate_limited_rate=0.0,
                client_cpu_fraction=0.3,
                server_container_cpu_util=0.9,
            )
        ],
        limits,
    )
    with pytest.raises(AssertionError, match="client, not the instance"):
        assert_measurement_valid([_result(client_cpu_fraction=0.99)], limits)
    with pytest.raises(AssertionError, match="not saturated"):
        assert_measurement_valid([_result(server_container_cpu_util=0.6)], limits)
    with pytest.raises(AssertionError, match="overloaded"):
        assert_measurement_valid([_result(rate_limited_rate=0.05)], limits)
    # Unknown evidence is not a failure.
    assert_measurement_valid([_result()], limits)
