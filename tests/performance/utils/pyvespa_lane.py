# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Spawn workers per transport to avoid a GIL bottleneck; keep this module free of pytest."""

import asyncio
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from typing import Callable, List, Optional, Tuple

from vespa.application import Vespa, VespaSync
from vespa.retries import NO_RETRY

from utils.metrics import LaneResult, percentiles
from utils.cpu_probes import LoadGeneratorCpu, InstanceCpuSampler
from utils.config import (
    ASYNC_HTTP_MODE,
    CONTAINER_CLUSTER,
    CONTENT_CLUSTER,
    SCHEMA,
    SYNC_HTTP_MODE,
    LoadProfile,
    make_doc,
)

BATCH_METHODS = ("feed_iterable", "feed_async_iterable")
HTTP_MODE = {
    "sync_feed_data_point": SYNC_HTTP_MODE,
    "async_feed_data_point": ASYNC_HTTP_MODE,
    "feed_iterable": SYNC_HTTP_MODE,
    "feed_async_iterable": ASYNC_HTTP_MODE,
}
# (completed_at as time.time(), HTTP status or 0 for no response, latency ms or None)
Completion = Tuple[float, int, Optional[float]]


@dataclass(frozen=True)
class Target:
    """Picklable connection details; each worker builds its own client."""

    transport: str  # "token" | "mtls"
    url: str
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    token: Optional[str] = field(default=None, repr=False)

    def app(self) -> Vespa:
        # Match k6: httpr otherwise requests compressed responses.
        headers = {"Accept-Encoding": "identity"}
        if self.transport == "mtls":
            return Vespa(
                url=self.url,
                cert=self.cert_path,
                key=self.key_path,
                additional_headers=headers,
            )
        return Vespa(
            url=self.url,
            vespa_cloud_secret_token=self.token,
            additional_headers=headers,
        )


@dataclass
class WorkerResult:
    """One worker process: its measurement window, CPU time over it, completions."""

    started: float  # time.time(); closed loops: end of warmup
    finished: float
    cpu_s: float
    completions: List[Completion]


def _status(response, error: Optional[BaseException]) -> int:
    """HTTP status of a response, or of the response behind a raised sync error."""
    if error is None:
        return response.status_code
    for _ in range(5):
        status = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
        error = error.__cause__ or error.__context__
        if error is None:
            break
    return 0


def sync_feed_data_point(target: Target, profile: LoadProfile, worker: int):
    """VespaSync.feed_data_point, `profile.concurrency` threads, closed loop."""
    app, prefix = target.app(), f"sfdp-{target.transport}-{worker}"
    warmup_end = time.time() + profile.warmup_s
    deadline = warmup_end + profile.duration_s

    def loop(sync_app: VespaSync) -> List[Completion]:
        completions: List[Completion] = []
        while time.time() < deadline:
            doc_id, fields = make_doc(prefix)
            begin = time.perf_counter()
            try:
                response, error = sync_app.feed_data_point(SCHEMA, doc_id, fields), None
            except Exception as e:
                response, error = None, e
            latency = (time.perf_counter() - begin) * 1000
            completions.append((time.time(), _status(response, error), latency))
        return completions

    # Share one multiplexed connection, as feed_iterable does.
    with VespaSync(
        app=app,
        pool_connections=profile.concurrency,
        pool_maxsize=profile.concurrency,
        compress=False,
        num_retries_429=0,
    ) as sync_app:
        with ThreadPoolExecutor(max_workers=profile.concurrency) as executor:
            futures = [
                executor.submit(loop, sync_app) for _ in range(profile.concurrency)
            ]
            time.sleep(max(0.0, warmup_end - time.time()))
            cpu_start = time.process_time()
            completions = [c for f in futures for c in f.result()]
            cpu_s = time.process_time() - cpu_start
    return WorkerResult(warmup_end, deadline, cpu_s, completions)


def async_feed_data_point(target: Target, profile: LoadProfile, worker: int):
    """VespaAsync.feed_data_point, `profile.concurrency` coroutines, closed loop."""
    app, prefix = target.app(), f"afdp-{target.transport}-{worker}"

    async def run() -> WorkerResult:
        completions: List[Completion] = []
        async with app.asyncio(
            connections=profile.async_connections,
            compress=False,
            docv1_retry_policy=NO_RETRY,
        ) as session:
            warmup_end = time.time() + profile.warmup_s
            deadline = warmup_end + profile.duration_s

            async def loop() -> None:
                while time.time() < deadline:
                    doc_id, fields = make_doc(prefix)
                    begin = time.perf_counter()
                    try:
                        response = await session.feed_data_point(
                            schema=SCHEMA, data_id=doc_id, fields=fields
                        )
                        error = None
                    except Exception as e:
                        response, error = None, e
                    latency = (time.perf_counter() - begin) * 1000
                    completions.append((time.time(), _status(response, error), latency))

            loops = asyncio.gather(*(loop() for _ in range(profile.concurrency)))
            await asyncio.sleep(max(0.0, warmup_end - time.time()))
            cpu_start = time.process_time()
            await loops
            return WorkerResult(
                warmup_end, deadline, time.process_time() - cpu_start, completions
            )

    return asyncio.run(run())


def _iterable(target: Target, profile: LoadProfile, worker: int, method: str):
    """Batch APIs expose completion times and statuses, but no per-request latency."""
    app, prefix = target.app(), f"{method}-{target.transport}-{worker}"
    completions: List[Completion] = []

    def make_batch(count: int) -> List[dict]:
        return [
            {"id": d, "fields": f} for d, f in (make_doc(prefix) for _ in range(count))
        ]

    def feed(docs: List[dict], callback) -> None:
        if method == "feed_iterable":
            # Bound queue scanning overhead so the runner does not become CPU-bound.
            app.feed_iterable(
                docs,
                schema=SCHEMA,
                callback=callback,
                max_workers=profile.concurrency,
                max_connections=profile.concurrency,
                max_queue_size=profile.concurrency,
                compress=False,
                num_retries_429=0,
            )
        else:
            app.feed_async_iterable(
                docs,
                schema=SCHEMA,
                callback=callback,
                max_workers=profile.concurrency,
                max_connections=profile.async_connections,
                max_queue_size=profile.concurrency,
                docv1_retry_policy=NO_RETRY,
            )

    # Untimed warmup batch: connection/TLS setup stays out of the measurement.
    feed(make_batch(profile.iterable_warmup_docs), lambda response, doc_id: None)
    docs = make_batch(profile.iterable_docs)
    started, cpu_start = time.time(), time.process_time()
    feed(docs, lambda r, doc_id: completions.append((time.time(), r.status_code, None)))
    cpu_s = time.process_time() - cpu_start
    # Documents without a callback count as failed.
    completions += [(time.time(), 0, None)] * (len(docs) - len(completions))
    return WorkerResult(started, time.time(), cpu_s, completions)


def feed_iterable(target: Target, profile: LoadProfile, worker: int):
    return _iterable(target, profile, worker, "feed_iterable")


def feed_async_iterable(target: Target, profile: LoadProfile, worker: int):
    return _iterable(target, profile, worker, "feed_async_iterable")


WORKERS: dict = {
    "sync_feed_data_point": sync_feed_data_point,
    "async_feed_data_point": async_feed_data_point,
    "feed_iterable": feed_iterable,
    "feed_async_iterable": feed_async_iterable,
}


def aggregate(
    method: str, transport: str, parts: List[WorkerResult], profile: LoadProfile
) -> LaneResult:
    """Merge the worker processes of one transport into a LaneResult."""
    if method in BATCH_METHODS:
        # Only count while every process is feeding, so startup and the tail
        # where stragglers run at reduced concurrency are excluded.
        start, end = max(p.started for p in parts), min(p.finished for p in parts)
        if end <= start:  # pathological skew: fall back to the whole span
            start, end = min(p.started for p in parts), max(p.finished for p in parts)
        window = [c for p in parts for c in p.completions if start <= c[0] <= end]
        duration_s = end - start
    else:
        # Each process counted completions inside its own window of duration_s.
        window = [
            c for p in parts for c in p.completions if p.started <= c[0] <= p.finished
        ]
        duration_s = profile.duration_s
    statuses = [status for _, status, _ in window]
    latencies = [latency for _, _, latency in window if latency is not None]
    p50, p95, p99 = percentiles(latencies) if latencies else (None, None, None)
    mean_ms = sum(latencies) / len(latencies) if latencies else None
    requests = len(window)
    rps = requests / duration_s if duration_s > 0 else 0.0
    cpu_s = sum(p.cpu_s for p in parts)
    return LaneResult(
        lane="pyvespa",
        method=method,
        transport=transport,
        http=HTTP_MODE[method],
        rps=rps,
        error_rate=sum(s != 200 for s in statuses) / requests if requests else 1.0,
        requests=requests,
        duration_s=duration_s,
        concurrency=profile.concurrency,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        mean_ms=mean_ms,
        achieved_in_flight=rps * mean_ms / 1000 if mean_ms is not None else None,
        processes=profile.processes,
        connections=profile.connections(),
        cpu_ms_per_request=cpu_s * 1000 / requests if requests else None,
        rate_limited_rate=statuses.count(429) / requests if requests else None,
        status_counts=dict(Counter(str(s) if s else "error" for s in statuses)),
        # Fallback when /proc/stat is unavailable: our processes' CPU over the
        # window as a share of the machine.
        client_cpu_fraction=(
            cpu_s / ((os.cpu_count() or 1) * duration_s) if duration_s > 0 else None
        ),
    )


def run_method(
    method: str,
    targets: List[Target],
    profile: LoadProfile,
    metrics_app: Optional[Vespa] = None,
) -> List[LaneResult]:
    """Run all targets concurrently, then merge workers and attach CPU measurements."""
    worker_fn: Callable = WORKERS[method]
    share = profile.per_process()
    load_start = time.time()
    runner_cpu = LoadGeneratorCpu().start()
    sampler = (
        InstanceCpuSampler(metrics_app).start() if metrics_app is not None else None
    )
    # spawn: no inherited threads or sockets, and the default on every platform
    # from Python 3.14. Children import this module via the parent's sys.path.
    with ProcessPoolExecutor(
        max_workers=len(targets) * profile.processes, mp_context=get_context("spawn")
    ) as executor:
        futures = {
            target.transport: [
                executor.submit(worker_fn, target, share, worker)
                for worker in range(profile.processes)
            ]
            for target in targets
        }
        parts = {
            transport: [f.result() for f in fs] for transport, fs in futures.items()
        }
    runner_fraction = runner_cpu.stop()
    server = sampler.stop(load_start) if sampler is not None else {}

    results = []
    for target in targets:
        result = aggregate(method, target.transport, parts[target.transport], profile)
        results.append(
            replace(
                result,
                client_cpu_fraction=(
                    runner_fraction
                    if runner_fraction is not None
                    else result.client_cpu_fraction
                ),
                server_container_cpu_util=server.get(f"container/{CONTAINER_CLUSTER}"),
                server_content_cpu_util=server.get(f"content/{CONTENT_CLUSTER}"),
            )
        )
    return results
