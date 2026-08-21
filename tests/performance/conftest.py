# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import pytest

from vespa.deployment import VespaCloud


# Persistent prod performance app, deployed once via
# test_deploy_performance_instance.py.
TENANT = "vespa-team"
APPLICATION = "pyvespa-performance"
INSTANCE = "default"
ENVIRONMENT = "prod"
REGION = "aws-us-east-1c"
SCHEMA = "msmarco"
CONTENT_CLUSTER = "msmarco_content"


@dataclass(frozen=True)
class PerformanceEndpoints:
    """Connection details for the k6 tooling.

    The token is excluded from repr so pytest failure output (which prints
    fixture values) never contains the secret.
    """

    mtls_url: str
    token_url: str
    cert_path: str
    key_path: str
    token: str = field(repr=False)


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"{name} must be set for performance tests.", pytrace=False)
    return value


@pytest.fixture(scope="session")
def vespa_cloud_token_endpoints() -> Generator[PerformanceEndpoints, None, None]:
    """
    Connect to the already-deployed, persistent prod performance instance and
    yield its mTLS + token endpoint details for the k6 tooling.

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

    app = vespa_cloud.get_application(
        instance=INSTANCE,
        environment=ENVIRONMENT,
        endpoint_type="mtls",
        region=REGION,
    )

    # Pre-clean leftovers from any earlier run that was killed before teardown.
    print("\n=== Setup: deleting any leftover test documents ===")
    app.delete_all_docs(content_cluster_name=CONTENT_CLUSTER, schema=SCHEMA)
    print("Leftover documents deleted.")

    try:
        yield PerformanceEndpoints(
            mtls_url=mtls_url,
            token_url=token_url,
            cert_path=str(cert_path),
            key_path=str(key_path),
            token=secret_token,
        )
    finally:
        # The k6 workload feeds docs, so leave a clean slate for the next run.
        print("\n=== Teardown: deleting fed test documents ===")
        app.delete_all_docs(content_cluster_name=CONTENT_CLUSTER, schema=SCHEMA)
        print("Fed documents deleted.")
