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

from utils.workloads import (
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
    """Connection details for both performance lanes (k6 and pyvespa).

    The token and app objects are excluded from repr so pytest failure
    output (which prints fixture values) never contains the secret.
    """

    mtls_url: str
    token_url: str
    cert_path: str
    key_path: str
    token: str = field(repr=False)
    vespa_cloud: VespaCloud = field(repr=False)
    mtls_app: Vespa = field(repr=False)
    token_app: Vespa = field(repr=False)
    # Effective load profile for this session (concurrency from the measured
    # ceiling and RTT, see LoadProfile.for_session); PROFILE if not measured.
    profile: LoadProfile = PROFILE


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"{name} must be set for performance tests.", pytrace=False)
    return value


@pytest.fixture(scope="session")
def vespa_cloud_token_endpoints() -> Generator[PerformanceEndpoints, None, None]:
    """
    Connect to the already-deployed, persistent prod performance instance and
    yield its mTLS + token endpoint details for both performance lanes.

    This does NOT deploy: the instance is created once by
    test_deploy_performance_instance.py and reused across runs for stable, comparable
    regression numbers. Endpoint lookups are read-only control-plane calls.

    Requires the environment variables VESPA_TEAM_API_KEY (control plane) and
    VESPA_CLOUD_SECRET_TOKEN (token data plane). mTLS additionally needs the
    data-plane cert/key pair locally (written by the deploy step into
    ~/.vespa/{tenant}.{app}.{instance}/); if absent, the test is skipped.
    """

    api_key = _require_env_var("VESPA_TEAM_API_KEY")
    secret_token = _require_env_var("VESPA_CLOUD_SECRET_TOKEN")

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

    # Control-plane connection only (no deploy). VespaCloud requires
    # application_package or application_root even for read-only endpoint
    # lookups, so pass a placeholder root -- get_*_endpoint hit the control-plane
    # API and never read it. The constructor loads the data-plane cert pair from
    # ~/.vespa/{tenant}.{app}.{instance}/ (a ./.vespa directory in the cwd would
    # take precedence) and, as a side effect, updates the global vespa CLI
    # config to point at this application.
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

    # Warm the instance (JIT, caches, connections) before the first measured
    # test, so whichever lane runs first is not penalized for finding a cold
    # container, and use the warmup's throughput as this session's ceiling
    # estimate. Together with the network RTT it sets the concurrency that
    # puts LoadProfile.server_queue_target requests inside the instance
    # regardless of where the load generator runs.
    if shutil.which("k6") is not None:
        from utils.k6_lane import run_k6

        print(f"\n=== Warmup: k6 for {int(WARMUP.warmup_s + WARMUP.duration_s)}s ===")
        warm = run_k6(
            endpoints,
            WARMUP,
            Path(os.environ.get("PERFORMANCE_REPORT_DIR") or ".") / "k6_warmup.json",
            extra_env=None,
        )
        mtls_app.delete_all_docs(
            content_cluster_name=CONTENT_CLUSTER, schema=SCHEMA, slices=CLEANUP_SLICES
        )
        print("Warmup documents deleted.")
        ceiling = sum(r.rps for r in warm)
        rtt = _network_rtt_s(mtls_app)
        profile = PROFILE.for_session(ceiling_rps=ceiling, rtt_s=rtt)
        print(
            f"Session profile: ceiling ~{ceiling:.0f} rps, RTT {rtt * 1000:.0f} ms -> "
            f"concurrency {profile.concurrency} per transport "
            f"({profile.connections()} connections x "
            f"{profile.streams_per_connection()} streams), "
            f"~{PROFILE.server_queue_target} queued in the instance"
        )
        endpoints = replace(endpoints, profile=profile)

    try:
        yield endpoints
    finally:
        # The workloads feed docs, so leave a clean slate for the next run.
        print("\n=== Teardown: deleting fed test documents ===")
        mtls_app.delete_all_docs(
            content_cluster_name=CONTENT_CLUSTER, schema=SCHEMA, slices=CLEANUP_SLICES
        )
        print("Fed documents deleted.")


def _network_rtt_s(app, samples: int = 20) -> float:
    """Round trip to the endpoint: the minimum of `samples` sequential tiny
    GETs on one warm connection (network plus a negligible handler)."""
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
    """Block until the container and content nodes are quiet (or max_wait_s),
    so background work left by the previous test or cleanup does not bleed
    into the next measurement. The metrics proxy refreshes about once a minute."""
    from utils.saturation import server_cpu_util

    deadline = time.time() + max_wait_s
    while True:
        util, _ = server_cpu_util(app)
        busiest = max(util.values()) if util else 0.0
        if busiest <= IDLE_CPU_UTIL or time.time() >= deadline:
            print(
                f"Instance CPU {busiest * 100:.0f}% (idle <= {IDLE_CPU_UTIL * 100:.0f}%)."
            )
            return
        print(f"Instance CPU {busiest * 100:.0f}%, waiting for it to settle...")
        time.sleep(15)


@pytest.fixture(autouse=True)
def settled_instance(vespa_cloud_token_endpoints):
    """Start every test on a quiet instance. Deleting the previous test's
    documents between tests (PERFORMANCE_CLEAN_BETWEEN_TESTS=1) is opt-in:
    removing ~600k documents per test left the content node busy for minutes
    (compaction, tombstone pruning) and halved a later test's throughput, a
    bigger state change than letting the corpus grow within a session."""
    _wait_until_instance_idle(vespa_cloud_token_endpoints.mtls_app)
    yield
    if os.environ.get("PERFORMANCE_CLEAN_BETWEEN_TESTS") == "1":
        print("\n=== Cleanup: deleting documents fed by this test ===")
        vespa_cloud_token_endpoints.mtls_app.delete_all_docs(
            content_cluster_name=CONTENT_CLUSTER, schema=SCHEMA, slices=CLEANUP_SLICES
        )
        print("Documents deleted.")


@pytest.fixture(scope="session")
def run_state() -> Dict:
    """Results shared across tests in one session, e.g. the opening k6 run so
    the closing k6 run can report how much the instance drifted meanwhile."""
    return {}


def pytest_collection_modifyitems(items):
    """Run tests marked `perf_last` after everything else (k6 first and last
    brackets the pyvespa methods, so instance drift within the run is visible)."""
    items.sort(key=lambda item: 1 if item.get_closest_marker("perf_last") else 0)
