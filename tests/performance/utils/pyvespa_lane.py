# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import time
from collections import Counter
from typing import Callable, List, Optional

from vespa.application import Vespa
from vespa.retries import NO_RETRY

from utils.config import (
    CONTAINER_CLUSTER,
    CONTENT_CLUSTER,
    SCHEMA,
    SMALL,
    FeedCase,
    LoadProfile,
    make_doc,
)
from utils.cpu_probes import InstanceCpuSampler, LoadGeneratorCpu
from utils.metrics import LaneResult


def client(
    transport: str,
    url: str,
    token: Optional[str] = None,
    cert_path: Optional[str] = None,
    key_path: Optional[str] = None,
) -> Vespa:
    """A Vespa client that, like k6, asks for uncompressed responses."""
    headers = {"Accept-Encoding": "identity"}
    if transport == "mtls":
        return Vespa(url=url, cert=cert_path, key=key_path, additional_headers=headers)
    return Vespa(url=url, vespa_cloud_secret_token=token, additional_headers=headers)


def _docs(prefix: str, count: int, body: str) -> List[dict]:
    return [
        {"id": d, "fields": f}
        for d, f in (make_doc(prefix, body) for _ in range(count))
    ]


def _feed(
    app: Vespa,
    method: str,
    docs: List[dict],
    callback: Callable,
    workers: int,
    compress: bool,
) -> None:
    """One call, retries off so every 429 is seen as such."""
    if method == "feed_iterable":
        app.feed_iterable(
            docs,
            schema=SCHEMA,
            callback=callback,
            max_workers=workers,
            compress=compress,
            num_retries_429=0,
        )
    else:
        assert not compress, "feed_async_iterable has no compression parameter"
        app.feed_async_iterable(
            docs,
            schema=SCHEMA,
            callback=callback,
            max_workers=workers,
            docv1_retry_policy=NO_RETRY,
        )


def run_pyvespa(
    method: str,
    app: Vespa,
    transport: str,
    profile: LoadProfile,
    metrics_app: Optional[Vespa] = None,
    case: FeedCase = SMALL,
) -> LaneResult:
    """Feed one batch through `method` and measure it whole. Returns the same
    LaneResult shape as the k6 lane, which the tests assert on and export to Prometheus."""
    prefix = f"{method}{case.suffix}-{transport}"
    workers = profile.pyvespa_workers
    # Untimed warmup batch: connection and TLS setup stay out of the measurement.
    warmup = _docs(prefix, profile.iterable_warmup_docs, case.body)
    _feed(app, method, warmup, lambda r, i: None, workers, case.gzip)

    docs = _docs(prefix, case.docs or profile.iterable_docs, case.body)
    statuses: List[int] = []
    runner_cpu = LoadGeneratorCpu().start()
    sampler = InstanceCpuSampler(metrics_app).start() if metrics_app else None
    started, cpu_start = time.time(), time.process_time()
    _feed(
        app,
        method,
        docs,
        lambda r, i: statuses.append(r.status_code),
        workers,
        case.gzip,
    )
    duration_s = time.time() - started
    cpu_s = time.process_time() - cpu_start
    runner_fraction = runner_cpu.stop()
    server = sampler.stop(started) if sampler else {}

    # Documents without a callback count as failed.
    statuses += [0] * (len(docs) - len(statuses))
    requests = len(statuses)
    return LaneResult(
        lane="pyvespa",
        method=method + case.suffix,
        transport=transport,
        rps=requests / duration_s,
        error_rate=sum(not 200 <= s < 300 for s in statuses) / requests,
        requests=requests,
        duration_s=duration_s,
        concurrency=profile.pyvespa_workers,
        cpu_ms_per_request=cpu_s * 1000 / requests,
        rate_limited_rate=statuses.count(429) / requests,
        client_cpu_fraction=runner_fraction,
        client_cpu_cores=cpu_s / duration_s,
        server_container_cpu_util=server.get(f"container/{CONTAINER_CLUSTER}"),
        server_content_cpu_util=server.get(f"content/{CONTENT_CLUSTER}"),
        server_cpu_samples=sampler.samples if sampler else None,
        status_counts=dict(Counter(str(s) if s else "error" for s in statuses)),
    )
