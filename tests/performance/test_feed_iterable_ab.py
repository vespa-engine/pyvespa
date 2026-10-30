# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Benchmark only: feed_iterable before and after the completion-event rewrite, in
one session on the same runner and instance, in A/B/B/A order per transport."""

import dataclasses
import math
import statistics

import pytest
from conftest import _wait_until_instance_idle

from utils.baseline_feed_iterable import BaselineVespa
from utils.config import SMALL
from utils.metrics import resolve_report_dir, write_records
from utils.pyvespa_lane import run_pyvespa
from vespa.application import Vespa

DOCS = 50_000  # per run; 16 runs feed what the regular pyvespa lane feeds
BLOCKS = 2
VARIANTS = {"baseline": BaselineVespa, "candidate": Vespa}
HEADERS = {"Accept-Encoding": "identity"}  # as utils.pyvespa_lane.client


def _client(cls, transport: str, endpoints):
    if transport == "mtls":
        return cls(
            url=endpoints.mtls_url,
            cert=endpoints.cert_path,
            key=endpoints.key_path,
            additional_headers=HEADERS,
        )
    return cls(
        url=endpoints.token_url,
        vespa_cloud_secret_token=endpoints.token,
        additional_headers=HEADERS,
    )


def _geomean(xs):
    return math.exp(statistics.mean(math.log(x) for x in xs))


@pytest.mark.performance
def test_feed_iterable_baseline_vs_candidate(vespa_cloud_token_endpoints, tmp_path):
    endpoints = vespa_cloud_token_endpoints
    case = dataclasses.replace(SMALL, docs=DOCS)
    profile = endpoints.profile_for(case)
    print(
        f"\n=== feed_iterable A/B: RTT {endpoints.rtt_s * 1000:.0f} ms, "
        f"max_workers {profile.pyvespa_workers}, {DOCS} docs per run ==="
    )
    results = []
    for transport in ("token", "mtls"):
        for block in range(BLOCKS):
            for name in ("baseline", "candidate", "candidate", "baseline"):
                _wait_until_instance_idle(endpoints.mtls_app, max_wait_s=60)
                app = _client(VARIANTS[name], transport, endpoints)
                result = run_pyvespa(
                    "feed_iterable", app, transport, profile, endpoints.mtls_app, case
                )
                result = dataclasses.replace(
                    result, method=f"feed_iterable_{name}_b{block}"
                )
                results.append(result)
                print(
                    f"{transport} block {block} {name}: {result.rps:.0f} rps, "
                    f"{result.cpu_ms_per_request:.3f} ms CPU/request, "
                    f"errors {result.error_rate:.2%}, server CPU container "
                    f"{result.server_container_cpu_util} content "
                    f"{result.server_content_cpu_util}"
                )
    write_records(results, resolve_report_dir(tmp_path), "pyvespa_feed_iterable_ab")

    print("\n=== Candidate / baseline per block (geometric means) ===")
    for transport in ("token", "mtls"):
        for block in range(BLOCKS):

            def ratio(field):
                def runs(name):
                    return [
                        getattr(r, field)
                        for r in results
                        if r.transport == transport
                        and r.method == f"feed_iterable_{name}_b{block}"
                    ]

                return _geomean(runs("candidate")) / _geomean(runs("baseline"))

            print(
                f"{transport} block {block}: rps x{ratio('rps'):.2f}, "
                f"CPU/request x{ratio('cpu_ms_per_request'):.2f}"
            )
    assert all(r.error_rate <= 0.02 for r in results)
