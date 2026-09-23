# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Multi-process load generation for the pyvespa lane.

One Python process is GIL-bound at roughly 2000 feed requests/s on a hosted
runner, below what the performance instance can absorb, so a single process
would measure the client and the runner's CPU rather than Vespa. Each method is
therefore run in `LoadProfile.processes` worker processes per transport, each
owning a share of the concurrency, and the samples are merged here.

Client behaviour is pinned to match the k6 script: no retries (a 429 is a
failed, rate-limited request on both sides), no compression, 120 s timeout.

This module is imported by spawned worker processes: keep it free of pytest.
"""

import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from typing import Callable, Dict, List, Optional, Tuple

from vespa.application import Vespa, VespaSync
from vespa.retries import NO_RETRY

from utils.metrics import LaneResult, percentiles
from utils.saturation import RunnerCpu, ServerCpuSampler
from utils.workloads import (
    ASYNC_HTTP_MODE,
    CONTAINER_CLUSTER,
    CONTENT_CLUSTER,
    SCHEMA,
    SYNC_HTTP_MODE,
    LoadProfile,
    make_doc,
)


@dataclass(frozen=True)
class Target:
    """Connection details for one transport, picklable for worker processes.

    Vespa objects are built inside the worker (mirrors
    VespaCloud.get_application) so no client state crosses the process
    boundary. The token is excluded from repr.
    """

    transport: str  # "token" | "mtls"
    url: str
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    token: Optional[str] = field(default=None, repr=False)

    def app(self) -> Vespa:
        # k6 sends no Accept-Encoding, so its responses come back uncompressed;
        # httpr asks for zstd/gzip/deflate/br by default and the container
        # gzips every response for it. Pin identity so both lanes cost the
        # instance the same per request. (pyvespa's default is compression on.)
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
    """What one worker process measured inside its window."""

    requests: int
    errors: int
    rate_limited: int  # responses with status 429 (also counted as errors)
    latencies_ms: List[float]  # empty for the iterable (batch) methods
    started: float  # time.time(), comparable across processes
    finished: float
    cpu_s: float  # process CPU time over the measurement window
    # Iterable (batch) methods only: (completed_at time.time(), status) per
    # document, so the parent can count completions inside the window where
    # every worker process was still feeding.
    completions: List[Tuple[float, int]] = field(default_factory=list)
    # status (or "error") -> count over the measured requests
    status_counts: Dict[str, int] = field(default_factory=dict)


def _windows(profile: LoadProfile):
    """(warmup_end, deadline) on perf_counter, measured from now."""
    start = time.perf_counter()
    return start + profile.warmup_s, start + profile.warmup_s + profile.duration_s


def _status_from_exception(error: BaseException) -> Optional[int]:
    """HTTP status behind a raised error, if any. pyvespa's sync paths raise on
    non-2xx (``raise_for_status``), with the response attached to the
    ``HTTPError`` that is the ``VespaError``'s cause."""
    seen = 0
    while error is not None and seen < 5:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
        error = error.__cause__ or error.__context__
        seen += 1
    return None


def _sample(begin: float, response, error: Optional[BaseException]) -> Dict:
    completed = time.perf_counter()
    if error is not None:
        status, ok = _status_from_exception(error), False
    else:
        status, ok = response.status_code, response.is_successful()
    return {
        "completed_at": completed,
        "latency_ms": (completed - begin) * 1000,
        "ok": ok,
        "status": status,
    }


def _closed_loop_result(
    samples: List[Dict], warmup_end: float, deadline: float, started: float, cpu_s
) -> WorkerResult:
    measured = [s for s in samples if warmup_end <= s["completed_at"] <= deadline]
    return WorkerResult(
        requests=len(measured),
        errors=sum(1 for s in measured if not s["ok"]),
        rate_limited=sum(1 for s in measured if s["status"] == 429),
        latencies_ms=[s["latency_ms"] for s in measured],
        started=started,
        finished=time.time(),
        cpu_s=cpu_s,
        status_counts=_count_statuses(s["status"] for s in measured),
    )


def _count_statuses(statuses) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for status in statuses:
        key = str(status) if status is not None else "error"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _sleep_until(perf_deadline: float) -> None:
    time.sleep(max(0.0, perf_deadline - time.perf_counter()))


def sync_feed_data_point(target: Target, profile: LoadProfile, worker: int):
    """VespaSync.feed_data_point, `profile.concurrency` threads, closed loop."""
    app = target.app()
    prefix = f"sfdp-{target.transport}-{worker}"
    started = time.time()
    warmup_end, deadline = _windows(profile)

    def loop(sync_app: VespaSync) -> List[Dict]:
        samples: List[Dict] = []
        while time.perf_counter() < deadline:
            doc_id, fields = make_doc(prefix)
            begin = time.perf_counter()
            try:
                response = sync_app.feed_data_point(
                    schema=SCHEMA, data_id=doc_id, fields=fields
                )
                samples.append(_sample(begin, response, error=None))
            except Exception as e:
                samples.append(_sample(begin, None, error=e))
        return samples

    # One pooled client shared by the threads (as feed_iterable does), one
    # connection per thread. A client per thread, each with its own runtime,
    # doubled p99 latency and left the instance at ~70% CPU at 400 threads.
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
            _sleep_until(warmup_end)
            cpu_start = time.process_time()
            per_thread = [f.result() for f in futures]
            cpu_s = time.process_time() - cpu_start
    samples = [s for thread in per_thread for s in thread]
    return _closed_loop_result(samples, warmup_end, deadline, started, cpu_s)


def async_feed_data_point(target: Target, profile: LoadProfile, worker: int):
    """VespaAsync.feed_data_point, `profile.concurrency` coroutines, closed loop."""
    app = target.app()
    prefix = f"afdp-{target.transport}-{worker}"
    started = time.time()

    async def run():
        samples: List[Dict] = []
        async with app.asyncio(
            connections=profile.async_connections,
            compress=False,
            docv1_retry_policy=NO_RETRY,
        ) as session:
            warmup_end, deadline = _windows(profile)

            async def loop() -> None:
                while time.perf_counter() < deadline:
                    doc_id, fields = make_doc(prefix)
                    begin = time.perf_counter()
                    try:
                        response = await session.feed_data_point(
                            schema=SCHEMA, data_id=doc_id, fields=fields
                        )
                        samples.append(_sample(begin, response, error=None))
                    except Exception as e:
                        samples.append(_sample(begin, None, error=e))

            loops = asyncio.gather(*(loop() for _ in range(profile.concurrency)))
            await asyncio.sleep(max(0.0, warmup_end - time.perf_counter()))
            cpu_start = time.process_time()
            await loops
            cpu_s = time.process_time() - cpu_start
            return samples, warmup_end, deadline, cpu_s

    samples, warmup_end, deadline, cpu_s = asyncio.run(run())
    return _closed_loop_result(samples, warmup_end, deadline, started, cpu_s)


def _iterable(target: Target, profile: LoadProfile, worker: int, method: str):
    """feed_iterable / feed_async_iterable own their loop and clients, so only
    completion times and statuses (from the callback) and CPU are measurable:
    throughput, error and 429 rates, no per-request latency."""
    app = target.app()
    prefix = f"{method}-{target.transport}-{worker}"

    def make_batch(count: int) -> List[Dict]:
        return [
            {"id": doc_id, "fields": fields}
            for doc_id, fields in (make_doc(prefix) for _ in range(count))
        ]

    completions: List[Tuple[float, int]] = []

    def callback(response, doc_id: str) -> None:
        completions.append((time.time(), response.status_code))

    def feed(docs: List[Dict], cb) -> None:
        if method == "feed_iterable":
            # max_queue_size bounds the futures the consumer keeps rescanning
            # (2 x max_queue_size); the default 1000 made this path 4x the
            # client CPU of the others and CPU-bound on a 4-vCPU runner.
            app.feed_iterable(
                docs,
                schema=SCHEMA,
                callback=cb,
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
                callback=cb,
                max_workers=profile.concurrency,
                max_connections=profile.async_connections,
                max_queue_size=profile.concurrency,
                docv1_retry_policy=NO_RETRY,
            )

    # Untimed warmup batch: connection/TLS setup stays out of the measurement.
    feed(make_batch(profile.iterable_warmup_docs), lambda response, doc_id: None)

    docs = make_batch(profile.iterable_docs)
    started = time.time()
    cpu_start = time.process_time()
    feed(docs, callback)
    cpu_s = time.process_time() - cpu_start
    return WorkerResult(
        requests=len(docs),
        errors=sum(1 for _, s in completions if s != 200)
        + (len(docs) - len(completions)),
        rate_limited=sum(1 for _, s in completions if s == 429),
        latencies_ms=[],
        started=started,
        finished=time.time(),
        cpu_s=cpu_s,
        completions=completions,
        status_counts=_count_statuses(
            [status for _, status in completions]
            + [None] * (len(docs) - len(completions))
        ),
    )


def feed_iterable(target: Target, profile: LoadProfile, worker: int):
    return _iterable(target, profile, worker, "feed_iterable")


def feed_async_iterable(target: Target, profile: LoadProfile, worker: int):
    return _iterable(target, profile, worker, "feed_async_iterable")


WORKERS: Dict[str, Callable[[Target, LoadProfile, int], WorkerResult]] = {
    "sync_feed_data_point": sync_feed_data_point,
    "async_feed_data_point": async_feed_data_point,
    "feed_iterable": feed_iterable,
    "feed_async_iterable": feed_async_iterable,
}

HTTP_MODE = {
    "sync_feed_data_point": SYNC_HTTP_MODE,
    "async_feed_data_point": ASYNC_HTTP_MODE,
    "feed_iterable": SYNC_HTTP_MODE,
    "feed_async_iterable": ASYNC_HTTP_MODE,
}


def aggregate(
    method: str, transport: str, parts: List[WorkerResult], profile: LoadProfile
) -> LaneResult:
    """Merge the worker processes of one transport into a LaneResult."""
    cpu_s = sum(p.cpu_s for p in parts)
    latencies = [x for p in parts for x in p.latencies_ms]
    if latencies:
        # Closed loop: every process counted completions inside its own
        # `duration_s` window, so the exact aggregate rate is count / window.
        requests = sum(p.requests for p in parts)
        errors = sum(p.errors for p in parts)
        rate_limited = sum(p.rate_limited for p in parts)
        duration_s = profile.duration_s
        p50, p95, p99 = percentiles(latencies)
        status_counts: Dict[str, int] = {}
        for p in parts:
            for key, count in p.status_counts.items():
                status_counts[key] = status_counts.get(key, 0) + count
    else:
        # Batch APIs: the same definition as the closed loop. Count only the
        # completions inside the window where every process was feeding (from
        # the last process to start until the first to finish), so the ramp-up
        # and the tail where stragglers run at reduced concurrency are excluded.
        # time.time() is shared across processes.
        start = max(p.started for p in parts)
        end = min(p.finished for p in parts)
        if end <= start:  # pathological skew: fall back to the whole span
            start = min(p.started for p in parts)
            end = max(p.finished for p in parts)
        window = [
            status
            for p in parts
            for completed_at, status in p.completions
            if start <= completed_at <= end
        ]
        requests = len(window)
        errors = sum(1 for s in window if s != 200)
        rate_limited = sum(1 for s in window if s == 429)
        duration_s = end - start
        p50 = p95 = p99 = None
        status_counts = _count_statuses(window)
    return LaneResult(
        lane="pyvespa",
        method=method,
        transport=transport,
        http=HTTP_MODE[method],
        rps=requests / duration_s if duration_s > 0 else 0.0,
        error_rate=errors / requests if requests else 1.0,
        requests=requests,
        duration_s=duration_s,
        concurrency=profile.concurrency,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        processes=profile.processes,
        connections=profile.connections(),
        cpu_ms_per_request=cpu_s * 1000 / requests if requests else None,
        rate_limited_rate=rate_limited / requests if requests else None,
        status_counts=status_counts,
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
    """Run `method` for every target concurrently, each spread over
    `profile.processes` worker processes, and return one LaneResult per target
    in the given order, stamped with runner CPU and (if `metrics_app` is given)
    the instance's peak CPU utilization sampled during the run."""
    worker_fn = WORKERS[method]
    share = profile.per_process()
    load_start = time.time()
    runner_cpu = RunnerCpu().start()
    sampler = ServerCpuSampler(metrics_app).start() if metrics_app is not None else None
    # spawn: no inherited threads or sockets, and the default on every platform
    # from Python 3.14. Children import this module via the parent's sys.path.
    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(targets) * profile.processes, mp_context=context
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
