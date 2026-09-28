# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import time
from collections import Counter
from typing import Callable, List, Optional

from vespa.application import Vespa
from vespa.retries import NO_RETRY

from utils.config import (
    CONTAINER_CLUSTER,
    CONTENT_CLUSTER,
    FEED_ASYNC_ITERABLE_PARAMETERS,
    FEED_ITERABLE_PARAMETERS,
    SCHEMA,
    LoadProfile,
    make_doc,
)
from utils.cpu_probes import InstanceCpuSampler, LoadGeneratorCpu
from utils.metrics import LaneResult

PARAMETERS = {
    "feed_iterable": FEED_ITERABLE_PARAMETERS,
    "feed_async_iterable": FEED_ASYNC_ITERABLE_PARAMETERS,
}


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


def _docs(prefix: str, count: int) -> List[dict]:
    return [{"id": d, "fields": f} for d, f in (make_doc(prefix) for _ in range(count))]


def _feed(app: Vespa, method: str, docs: List[dict], callback: Callable) -> None:
    """One call, retries off so every 429 is seen as such."""
    if method == "feed_iterable":
        app.feed_iterable(
            docs,
            schema=SCHEMA,
            callback=callback,
            compress=False,
            num_retries_429=0,
            **PARAMETERS[method],
        )
    else:
        app.feed_async_iterable(
            docs,
            schema=SCHEMA,
            callback=callback,
            docv1_retry_policy=NO_RETRY,
            **PARAMETERS[method],
        )


def run_pyvespa(
    method: str,
    app: Vespa,
    transport: str,
    profile: LoadProfile,
    metrics_app: Optional[Vespa] = None,
) -> LaneResult:
    """Feed one batch through `method` and measure it whole. Returns the same
    LaneResult shape as the k6 lane, which the tests assert on and export to Prometheus."""
    prefix = f"{method}-{transport}"
    # Untimed warmup batch: connection and TLS setup stay out of the measurement.
    _feed(app, method, _docs(prefix, profile.iterable_warmup_docs), lambda r, i: None)

    docs = _docs(prefix, profile.iterable_docs)
    statuses: List[int] = []
    runner_cpu = LoadGeneratorCpu().start()
    sampler = InstanceCpuSampler(metrics_app).start() if metrics_app else None
    started, cpu_start = time.time(), time.process_time()
    _feed(app, method, docs, lambda r, i: statuses.append(r.status_code))
    duration_s = time.time() - started
    cpu_s = time.process_time() - cpu_start
    runner_fraction = runner_cpu.stop()
    server = sampler.stop(started) if sampler else {}

    # Documents without a callback count as failed.
    statuses += [0] * (len(docs) - len(statuses))
    requests = len(statuses)
    return LaneResult(
        lane="pyvespa",
        method=method,
        transport=transport,
        rps=requests / duration_s,
        error_rate=sum(s != 200 for s in statuses) / requests,
        requests=requests,
        duration_s=duration_s,
        concurrency=PARAMETERS[method]["max_workers"],
        cpu_ms_per_request=cpu_s * 1000 / requests,
        rate_limited_rate=statuses.count(429) / requests,
        client_cpu_fraction=runner_fraction,
        server_container_cpu_util=server.get(f"container/{CONTAINER_CLUSTER}"),
        server_content_cpu_util=server.get(f"content/{CONTENT_CLUSTER}"),
        status_counts=dict(Counter(str(s) if s else "error" for s in statuses)),
    )
