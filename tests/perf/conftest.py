# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
from typing import Dict, Generator

import pytest

from vespa.deployment import VespaCloud


# Persistent prod perf app, deployed once via test_deploy_perf_instance.py.
TENANT = "vespa-team"
APPLICATION = "pyvespa-performance"
INSTANCE = "default"
ENVIRONMENT = "prod"
REGION = "aws-euc1-az1"  # Frankfurt; closest prod zone to Norway.
SCHEMA = "msmarco"
CONTENT_CLUSTER = "msmarco_content"


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"{name} must be set for perf tests.", pytrace=False)
    return value


@pytest.fixture(scope="session")
def vespa_cloud_token_endpoints() -> Generator[Dict[str, str], None, None]:
    """
    Connect to the already-deployed, persistent prod perf instance and yield its
    mTLS + token endpoint details for the k6 tooling.

    This does NOT deploy: the instance is created once by
    test_deploy_perf_instance.py and reused across runs for stable, comparable
    regression numbers. Endpoint lookups are read-only control-plane calls.

    Requires the standard op env: VESPA_TEAM_API_KEY (control plane) and
    VESPA_CLOUD_SECRET_TOKEN (token data plane). mTLS additionally needs the
    data-plane cert/key pair locally (written by the deploy step into
    ~/.vespa/{tenant}.{app}.{instance}/); if absent, the test is skipped.
    """

    api_key = _require_env_var("VESPA_TEAM_API_KEY")
    secret_token = _require_env_var("VESPA_CLOUD_SECRET_TOKEN")

    # Control-plane connection only (no deploy). VespaCloud requires
    # application_package or application_root even for read-only endpoint
    # lookups, so pass a placeholder root -- get_*_endpoint hit the control-plane
    # API and never read it. The constructor loads the data-plane cert pair from
    # ~/.vespa/{tenant}.{app}.{instance}/, populating data_cert_path/data_key_path.
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

    cert_path = vespa_cloud.data_cert_path
    key_path = vespa_cloud.data_key_path
    if not cert_path or not key_path:
        pytest.skip(
            "mTLS certificate/key were not found locally "
            f"(~/.vespa/{TENANT}.{APPLICATION}.{INSTANCE}/). "
            "Deploy the perf instance from this machine first, or run "
            "'vespa auth cert'."
        )

    try:
        yield {
            "mtls_url": mtls_url,
            "token_url": token_url,
            "token": secret_token,
            "cert_path": cert_path,
            "key_path": key_path,
        }
    finally:
        # The k6 workload feeds docs, so leave a clean slate for the next run.
        app = vespa_cloud.get_application(
            instance=INSTANCE,
            environment=ENVIRONMENT,
            endpoint_type="mtls",
            region=REGION,
        )
        app.delete_all_docs(content_cluster_name=CONTENT_CLUSTER, schema=SCHEMA)
