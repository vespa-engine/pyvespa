# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Generator

import pytest

from vespa.deployment import VespaCloud
from vespa.application import Vespa, VespaSync

from utils.k6_lane import run_k6
from utils.metrics import resolve_report_dir
from utils.config import (
    APPLICATION,
    CLEANUP_SLICES,
    CONTENT_CLUSTER,
    IDLE_CPU_UTIL,
    PROFILE,
    WARMUP,
    LoadProfile,
    ENVIRONMENT,
    INSTANCE,
    REGION,
    SCHEMA,
    TENANT,
)


@dataclass(frozen=True)
class PerformanceEndpoints:
    """Connection details; credentials and clients are excluded from pytest repr."""

    mtls_url: str
    token_url: str
    cert_path: str
    key_path: str
    token: str = field(repr=False)
    vespa_cloud: VespaCloud = field(repr=False)
    mtls_app: Vespa = field(repr=False)
    token_app: Vespa = field(repr=False)
    # How many requests each lane keeps in flight depends on the round trip,
    # and GitHub's hosted runners land in different regions (4 to 67 ms from
    # the instance), so for_session sizes both lanes once per session. k6 must
    # keep the instance's queue full without overflowing into 429s: 200 queued
    # + ceiling × round trip. pyvespa, one GIL-bound process, gets enough in
    # flight to reach the ceiling if it could, ceiling × (round trip + service
    # time), so anything below the ceiling is client cost, not waiting.
    # PROFILE's defaults when the session could not measure.
    profile: LoadProfile = PROFILE


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"{name} must be set for performance tests.", pytrace=False)
    return value


@pytest.fixture(scope="session")
def vespa_cloud_token_endpoints(
    tmp_path_factory,
) -> Generator[PerformanceEndpoints, None, None]:
    """Connect to the persistent app, warm it, and clean up after the session."""

    api_key = _require_env_var("VESPA_TEAM_API_KEY")
    secret_token = _require_env_var("VESPA_CLOUD_SECRET_TOKEN")
    if shutil.which("k6") is None:
        pytest.fail("k6 is required: it measures the instance ceiling.", pytrace=False)

    # Check the mTLS cert pair up front: without one, VespaCloud would
    # auto-generate a fresh pair the deployed app does not authorize, and the
    # mTLS scenario would run doomed to 100% errors.
    cert_dir = Path.home() / ".vespa" / f"{TENANT}.{APPLICATION}.{INSTANCE}"
    cert_path = cert_dir / "data-plane-public-cert.pem"
    key_path = cert_dir / "data-plane-private-key.pem"
    if not cert_path.exists() or not key_path.exists():
        pytest.skip(
            f"mTLS certificate/key not found in {cert_dir}. Deploy the "
            "performance instance from this machine first, or run "
            "'vespa auth cert'."
        )

    # Endpoint lookup requires a placeholder application_root but does not deploy.
    vespa_cloud = VespaCloud(
        tenant=TENANT,
        application=APPLICATION,
        instance=INSTANCE,
        key_content=api_key.replace(r"\n", "\n"),
        application_root=".",
    )

    mtls_url = vespa_cloud.get_mtls_endpoint(
        instance=INSTANCE, environment=ENVIRONMENT, region=REGION
    )
    token_url = vespa_cloud.get_token_endpoint(
        instance=INSTANCE, environment=ENVIRONMENT, region=REGION
    )

    mtls_app = vespa_cloud.get_application(
        instance=INSTANCE,
        environment=ENVIRONMENT,
        endpoint_type="mtls",
        region=REGION,
    )

    token_app = vespa_cloud.get_application(
        instance=INSTANCE,
        environment=ENVIRONMENT,
        endpoint_type="token",
        vespa_cloud_secret_token=secret_token,
        region=REGION,
    )

    # Pre-clean leftovers from any earlier run that was killed before teardown.
    print("\n=== Setup: deleting any leftover test documents ===")
    mtls_app.delete_all_docs(
        content_cluster_name=CONTENT_CLUSTER, schema=SCHEMA, slices=CLEANUP_SLICES
    )
    print("Leftover documents deleted.")

    endpoints = PerformanceEndpoints(
        mtls_url=mtls_url,
        token_url=token_url,
        cert_path=str(cert_path),
        key_path=str(key_path),
        token=secret_token,
        vespa_cloud=vespa_cloud,
        mtls_app=mtls_app,
        token_app=token_app,
    )

    # Warmup throughput and network round trip set k6's concurrency (see `profile`).
    print(f"\n=== Warmup: k6 for {int(WARMUP.warmup_s + WARMUP.duration_s)}s ===")
    report_dir = resolve_report_dir(tmp_path_factory.mktemp("performance-report"))
    warm = run_k6(endpoints, WARMUP, report_dir / "k6_warmup.json", "mtls")
    mtls_app.delete_all_docs(
        content_cluster_name=CONTENT_CLUSTER, schema=SCHEMA, slices=CLEANUP_SLICES
    )
    print("Warmup documents deleted.")
    # Successful requests only: 429s are not capacity.
    ceiling = warm.rps * (1 - warm.error_rate)
    rtt = _network_rtt_s(mtls_app)
    profile = PROFILE.for_session(ceiling_rps=ceiling, rtt_s=rtt)
    print(
        f"Session profile: ceiling ~{ceiling:.0f} rps, RTT {rtt * 1000:.0f} ms -> "
        f"k6 concurrency {profile.concurrency} for the active transport "
        f"over {profile.k6_connections} connections, "
        f"~{PROFILE.server_queue_target} queued in the instance; "
        f"pyvespa max_workers {profile.pyvespa_workers}"
    )
    endpoints = replace(endpoints, profile=profile)

    try:
        yield endpoints
    finally:
        # Best effort: the next session's setup deletes whatever is left, and a
        # cleanup failure must not turn a passed measurement into an error.
        print("\n=== Teardown: deleting fed test documents ===")
        try:
            mtls_app.delete_all_docs(
                content_cluster_name=CONTENT_CLUSTER,
                schema=SCHEMA,
                slices=CLEANUP_SLICES,
            )
            print("Fed documents deleted.")
        except Exception as e:
            print(f"Teardown cleanup did not finish, setup will retry next run: {e}")


def _network_rtt_s(app, samples: int = 20) -> float:
    best = float("inf")
    with VespaSync(app=app, pool_connections=1, pool_maxsize=1) as session:
        url = f"{app.end_point}/ApplicationStatus"
        session.http_client.get(url, timeout=30)  # connection + TLS setup
        for _ in range(samples):
            started = time.perf_counter()
            session.http_client.get(url, timeout=30)
            best = min(best, time.perf_counter() - started)
    return best


def _wait_until_instance_idle(app, max_wait_s: float = 240.0) -> None:
    """Wait up to max_wait_s for background work from cleanup or the previous test."""
    from utils.cpu_probes import instance_cpu_util

    deadline = time.time() + max_wait_s
    while True:
        util, _ = instance_cpu_util(app)
        busiest = max(util.values()) if util else None
        label = f"{busiest:.0%}" if busiest is not None else "unknown"
        if busiest is not None and busiest <= IDLE_CPU_UTIL:
            print(f"Instance CPU {label} (idle <= {IDLE_CPU_UTIL:.0%}).")
            return
        if time.time() >= deadline:
            print(f"Instance CPU {label} after {max_wait_s:.0f} s; continuing anyway.")
            return
        print(f"Instance CPU {label}, waiting for it to settle...")
        time.sleep(15)


@pytest.fixture(autouse=True)
def settled_instance(request):
    """Before a performance test, wait for the previous one's background work to settle."""
    if request.node.get_closest_marker("performance"):
        endpoints = request.getfixturevalue("vespa_cloud_token_endpoints")
        _wait_until_instance_idle(endpoints.mtls_app)


@pytest.fixture(scope="session")
def run_state() -> Dict:
    """Share the opening k6 baseline with the remaining tests."""
    return {}


def pytest_collection_modifyitems(items):
    """Bracket the pyvespa tests with opening and closing k6 runs."""
    items.sort(key=lambda item: 1 if item.get_closest_marker("performance_last") else 0)
