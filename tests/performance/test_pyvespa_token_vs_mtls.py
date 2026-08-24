# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Tuple

import pytest

from vespa.application import Vespa, VespaSync

from utils.metrics import (
    LaneResult,
    assert_token_vs_mtls,
    resolve_report_dir,
    summarize_samples,
    write_records,
)
from utils.workloads import (
    ASYNC_HTTP_MODE,
    PYVESPA_THRESHOLDS,
    SCHEMA,
    SYNC_HTTP_MODE,
    LoadProfile,
    make_doc,
    method_profile,
)


def _closed_loop_windows(profile: LoadProfile) -> Tuple[float, float]:
    """Return (warmup_end, deadline) measured from now."""
    start = time.perf_counter()
    return start + profile.warmup_s, start + profile.warmup_s + profile.duration_s


def _measured(samples: List[Dict], warmup_end: float, deadline: float) -> List[Dict]:
    """Keep only completions inside the measurement window, so
    count/duration_s is an exact rate (in-flight tails are excluded)."""
    return [s for s in samples if warmup_end <= s["completed_at"] <= deadline]


def _result_from_samples(
    method: str,
    transport: str,
    http: str,
    samples: List[Dict],
    profile: LoadProfile,
) -> LaneResult:
    summary = summarize_samples(samples)
    return LaneResult(
        lane="pyvespa",
        method=method,
        transport=transport,
        http=http,
        rps=summary["requests"] / profile.duration_s,
        error_rate=summary["error_rate"],
        requests=summary["requests"],
        duration_s=profile.duration_s,
        concurrency=profile.concurrency,
        p50_ms=summary["p50_ms"],
        p95_ms=summary["p95_ms"],
        p99_ms=summary["p99_ms"],
    )


def _run_sync_closed_loop(
    app: Vespa, transport: str, profile: LoadProfile
) -> LaneResult:
    prefix = f"sfdp-{transport}"
    warmup_end, deadline = _closed_loop_windows(profile)

    def worker() -> List[Dict]:
        samples: List[Dict] = []
        with VespaSync(app=app) as sync_app:
            while time.perf_counter() < deadline:
                doc_id, fields = make_doc(prefix)
                started = time.perf_counter()
                try:
                    response = sync_app.feed_data_point(
                        schema=SCHEMA, data_id=doc_id, fields=fields
                    )
                    ok = response.is_successful()
                except Exception:
                    ok = False
                completed = time.perf_counter()
                samples.append(
                    {
                        "completed_at": completed,
                        "latency_ms": (completed - started) * 1000,
                        "ok": ok,
                    }
                )
        return samples

    with ThreadPoolExecutor(max_workers=profile.concurrency) as executor:
        per_worker = list(executor.map(lambda _: worker(), range(profile.concurrency)))
    samples = _measured(
        [s for worker in per_worker for s in worker], warmup_end, deadline
    )
    return _result_from_samples(
        "sync_feed_data_point", transport, SYNC_HTTP_MODE, samples, profile
    )


def _run_async_closed_loop(
    app: Vespa, transport: str, profile: LoadProfile
) -> LaneResult:
    prefix = f"afdp-{transport}"

    async def run() -> List[Dict]:
        samples: List[Dict] = []
        async with app.asyncio(connections=1) as session:
            warmup_end, deadline = _closed_loop_windows(profile)

            async def loop() -> None:
                while time.perf_counter() < deadline:
                    doc_id, fields = make_doc(prefix)
                    started = time.perf_counter()
                    try:
                        response = await session.feed_data_point(
                            schema=SCHEMA, data_id=doc_id, fields=fields
                        )
                        ok = response.is_successful()
                    except Exception:
                        ok = False
                    completed = time.perf_counter()
                    samples.append(
                        {
                            "completed_at": completed,
                            "latency_ms": (completed - started) * 1000,
                            "ok": ok,
                        }
                    )

            await asyncio.gather(*(loop() for _ in range(profile.concurrency)))
            return _measured(samples, warmup_end, deadline)

    samples = asyncio.run(run())
    return _result_from_samples(
        "async_feed_data_point", transport, ASYNC_HTTP_MODE, samples, profile
    )


def _run_iterable(
    app: Vespa, method: str, transport: str, profile: LoadProfile
) -> LaneResult:
    """feed_iterable / feed_async_iterable own their loop and clients, so only
    wall-clock throughput and callback-observed errors are measurable. These
    are batch APIs: throughput is the user-facing metric."""
    prefix = f"{method}-{transport}"
    http_mode = ASYNC_HTTP_MODE if method == "feed_async_iterable" else SYNC_HTTP_MODE

    def make_batch(count: int) -> List[Dict]:
        return [
            {"id": doc_id, "fields": fields}
            for doc_id, fields in (make_doc(prefix) for _ in range(count))
        ]

    failed: List[str] = []

    def callback(response, doc_id: str) -> None:
        if not response.is_successful():
            failed.append(doc_id)

    def feed(docs: List[Dict], cb) -> None:
        if method == "feed_iterable":
            app.feed_iterable(
                docs,
                schema=SCHEMA,
                callback=cb,
                max_workers=profile.iterable_max_workers,
                max_connections=profile.iterable_max_connections,
            )
        else:
            app.feed_async_iterable(
                docs,
                schema=SCHEMA,
                callback=cb,
                max_workers=profile.iterable_max_workers,
                max_connections=1,
            )

    # Untimed warmup batch: connection/TLS setup stays out of the measurement.
    feed(make_batch(profile.iterable_warmup_docs), lambda response, doc_id: None)

    docs = make_batch(profile.iterable_docs)
    started = time.perf_counter()
    feed(docs, callback)
    duration_s = time.perf_counter() - started

    return LaneResult(
        lane="pyvespa",
        method=method,
        transport=transport,
        http=http_mode,
        rps=len(docs) / duration_s,
        error_rate=len(failed) / len(docs),
        requests=len(docs),
        duration_s=duration_s,
        concurrency=profile.iterable_max_workers,
    )


def _run_pair(runner, endpoints, report_dir, method: str) -> None:
    profile = method_profile(method)
    expected_s = int(profile.warmup_s + profile.duration_s)
    print(
        f"\n=== Running pyvespa {method} "
        f"(concurrency={profile.concurrency}, ~{expected_s}s per transport) ==="
    )
    results = []
    for transport, app in [
        ("token", endpoints.token_app),
        ("mtls", endpoints.mtls_app),
    ]:
        print(f"--- {transport} ---")
        results.append(runner(app=app, transport=transport, profile=profile))
    write_records(results, report_dir, f"pyvespa_{method}")
    assert_token_vs_mtls(results[0], results[1], PYVESPA_THRESHOLDS)


@pytest.mark.performance
def test_sync_feed_data_point(vespa_cloud_token_endpoints, tmp_path):
    _run_pair(
        _run_sync_closed_loop,
        vespa_cloud_token_endpoints,
        resolve_report_dir(tmp_path),
        "sync_feed_data_point",
    )


@pytest.mark.performance
def test_async_feed_data_point(vespa_cloud_token_endpoints, tmp_path):
    _run_pair(
        _run_async_closed_loop,
        vespa_cloud_token_endpoints,
        resolve_report_dir(tmp_path),
        "async_feed_data_point",
    )


@pytest.mark.performance
def test_feed_iterable(vespa_cloud_token_endpoints, tmp_path):
    _run_pair(
        partial(_run_iterable, method="feed_iterable"),
        vespa_cloud_token_endpoints,
        resolve_report_dir(tmp_path),
        "feed_iterable",
    )


@pytest.mark.performance
def test_feed_async_iterable(vespa_cloud_token_endpoints, tmp_path):
    _run_pair(
        partial(_run_iterable, method="feed_async_iterable"),
        vespa_cloud_token_endpoints,
        resolve_report_dir(tmp_path),
        "feed_async_iterable",
    )
