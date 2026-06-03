# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import unittest
from datetime import datetime, timedelta

from vespa.package import (
    AuthClient,
    Parameter,
    DeploymentConfiguration,
    ContentCluster,
    ContainerCluster,
    Nodes,
    Validation,
    ValidationID,
)
from vespa.deployment import VespaCloud

from tests.integration.test_integration_docker import (
    create_msmarco_application_package,
)

# Reuse the existing tenant data-plane token (tenant+id scoped, so it works for
# this dedicated app too). The op env sets VESPA_CLIENT_TOKEN_ID; the deployed
# app declares <token id=...> with this value so VESPA_CLOUD_SECRET_TOKEN
# authenticates. Fallback matches the token used by the integration tests.
CLIENT_TOKEN_ID = os.environ.get("VESPA_CLIENT_TOKEN_ID", "pyvespa_integration")

# Closest prod zone to Norway that Vespa Cloud supports (Frankfurt, eu-central-1).
PERF_PROD_REGION = "aws-euc1-az1"


class TestDeployPerfInstanceToProd(unittest.TestCase):
    """Manual trigger for the persistent, perf-only prod instance.

    Mirrors TestMsmarcoProdApplicationWithTokenAuth: the deploy happens in
    setUp() and the test is @unittest.skip'd, so this NEVER runs in CI (skip is
    evaluated before setUp). Run it by hand when the perf app package changes:

        uv run pytest tests/perf/test_deploy_perf_instance.py -s -v

    Deploys vespa-team.pyvespa-performance.default to prod via the CD pipeline:
    a small, stable, exclusive instance used only for perf regression tracking.
    A dedicated application (not an instance of pyvespa-integration) so its
    deployment.xml is independent and perf work can never touch the integration
    apps. A separate app -- rather than a pyvespa-integration.perf-test instance
    -- is chosen for isolation: deployment.xml is application-wide, so adding a
    perf instance would mean re-declaring the existing default instance too
    (risking clobbering it). Note DeploymentConfiguration cannot emit <instance>
    blocks; the VT DSL (vespa.configuration.deployment.instance) can, but a
    dedicated app keeps "don't touch default" a structural guarantee.
    """

    def setUp(self) -> None:
        schema_name = "msmarco"
        self.auth_clients = [
            AuthClient(
                id="mtls",
                permissions=["read", "write"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="token",
                permissions=["read", "write"],
                parameters=[Parameter("token", {"id": CLIENT_TOKEN_ID})],
            ),
        ]
        self.app_package = create_msmarco_application_package(
            auth_clients=self.auth_clients
        )
        # Single small node per cluster: stable and cheap, enough for relative
        # regression tracking over time (not absolute capacity benchmarking).
        # m6g.large-equivalent (2 vcpu / 8Gb / arm64): the smallest non-burstable
        # AWS flavor, so CPU is fixed and week-over-week numbers are comparable.
        perf_resources = Parameter(
            "resources",
            {"vcpu": "2.0", "memory": "8Gb", "disk": "50Gb", "architecture": "arm64"},
        )
        self.app_package.clusters = [
            ContentCluster(
                id=f"{schema_name}_content",
                nodes=Nodes(count="1", parameters=[perf_resources]),
                document_name=schema_name,
                min_redundancy="1",
            ),
            ContainerCluster(
                id=f"{schema_name}_container",
                nodes=Nodes(count="1", parameters=[perf_resources]),
                auth_clients=self.auth_clients,
            ),
        ]
        self.app_package.deployment_config = DeploymentConfiguration(
            environment="prod", regions=[PERF_PROD_REGION]
        )
        # Single 1-node clusters in prod need two first-deployment overrides:
        # redundancy=1 (no HA replica) and minimum-node-count (<2 nodes). Both are
        # intentional for a small, stable perf box. minimum-node-count is not in
        # pyvespa's ValidationID enum, so pass it as a raw string.
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        self.app_package.validations = [
            Validation(ValidationID("redundancy-one"), tomorrow),
            Validation("minimum-node-count", tomorrow),
        ]

        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-performance",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_package=self.app_package,
            auth_client_token_id=CLIENT_TOKEN_ID,
        )
        self.instance_name = "default"
        self.build_no = self.vespa_cloud.deploy_to_prod(
            instance=self.instance_name,
            source_url="https://github.com/vespa-engine/pyvespa",
        )

    @unittest.skip(
        "Manual only. Deploys the persistent perf-test prod instance via the CD "
        "pipeline; too slow for CI. Remove the skip to run by hand."
    )
    def test_deploy_perf_instance(self):
        success = self.vespa_cloud.wait_for_prod_deployment(
            build_no=self.build_no, max_wait=3600
        )
        if not success:
            self.fail("Deployment failed")
