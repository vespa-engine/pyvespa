# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import shutil
from typing import Dict, Generator

import pytest

from vespa.deployment import VespaCloud
from vespa.package import AuthClient, Parameter

from tests.integration.test_integration_docker import create_msmarco_application_package


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"{name} must be set for perf tests.", pytrace=False)
    return value


@pytest.fixture(scope="session")
def vespa_cloud_token_endpoints(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Dict[str, str], None, None]:
    """
    Deploy a minimal Vespa Cloud app with both mTLS and token endpoints and yield the connection details.

    Mirrors the setup in integration token tests but exposes the URLs and credentials for perf tooling.
    """

    api_key = _require_env_var("VESPA_TEAM_API_KEY")
    secret_token = _require_env_var("VESPA_CLOUD_SECRET_TOKEN")
    tenant = _require_env_var("TENANT_NAME")
    application = "pyvespa-integration"
    instance_name = "perf-test"
    client_token_id = _require_env_var("VESPA_CLIENT_TOKEN_ID")

    disk_folder = tmp_path_factory.mktemp("vespa-perf-app")
    clients = [
        AuthClient(
            id="mtls",
            permissions=["read", "write"],
            parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
        ),
        AuthClient(
            id="token",
            permissions=["read", "write"],
            parameters=[Parameter("token", {"id": client_token_id})],
        ),
    ]
    app_package = create_msmarco_application_package(auth_clients=clients)

    vespa_cloud = VespaCloud(
        tenant=tenant,
        application=application,
        key_content=api_key.replace(r"\n", "\n"),
        application_package=app_package,
        auth_client_token_id=client_token_id,
    )

    vespa_cloud.deploy(instance=instance_name, disk_folder=str(disk_folder))
    token_app = vespa_cloud.get_application(
        instance=instance_name,
        environment="dev",
        endpoint_type="token",
        vespa_cloud_secret_token=secret_token,
    )

    mtls_url = vespa_cloud.get_mtls_endpoint(instance=instance_name, environment="dev")
    token_url = vespa_cloud.get_token_endpoint(instance=instance_name, environment="dev")

    cert_path = vespa_cloud.data_cert_path
    key_path = vespa_cloud.data_key_path
    if not cert_path or not key_path:
        vespa_cloud.delete(instance=instance_name)
        pytest.skip("mTLS certificate/key were not found; ensure vespa auth cert has been run.")

    try:
        yield {
            "mtls_url": mtls_url,
            "token_url": token_url,
            "token": secret_token,
            "cert_path": cert_path,
            "key_path": key_path,
        }
    finally:
        token_app.delete_all_docs(
            content_cluster_name="msmarco_content", schema=app_package.name
        )
        shutil.rmtree(disk_folder, ignore_errors=True)
        vespa_cloud.delete(instance=instance_name)
