import unittest
from tempfile import TemporaryDirectory
import os
from unittest.mock import patch, MagicMock

from vespa.deployment import VespaCloud


class TestVespaCloud(unittest.TestCase):
    def setUp(self):
        self.tenant = "test_tenant"
        self.application = "test_app"
        self.application_package = MagicMock()
        # Mock ._try_get_access_token to avoid requests
        VespaCloud._try_get_access_token = MagicMock(return_value="fake_access_token")

        self.vespa_cloud = VespaCloud(
            tenant=self.tenant,
            application=self.application,
            application_package=self.application_package,
        )

    @patch("vespa.deployment.VespaCloud._read_private_key")
    @patch("vespa.deployment.VespaCloud._load_certificate_pair")
    def test_initialization_no_disk_package(self, mock_load_cert, mock_read_key):
        mock_read_key.return_value = MagicMock()
        mock_load_cert.return_value = (MagicMock(), MagicMock())

        # This should raise ValueError since none of disk_folder and application_package are provided
        with self.assertRaises(ValueError):
            _vespa_cloud = VespaCloud(self.tenant, self.application)

    @patch("vespa.deployment.VespaCloud._request")
    def test_get_all_endpoints(self, mock_request):
        # Mock get_dev_region to avoid requests
        VespaCloud.get_dev_region = MagicMock(return_value="dev-region")

        mock_endpoints = [
            {"url": "https://endpoint1.vespa.oath.cloud:4443", "authMethod": "mtls"},
            {"url": "https://endpoint2.vespa.oath.cloud:4443", "authMethod": "token"},
        ]
        mock_request.return_value = {"endpoints": mock_endpoints}

        endpoints = self.vespa_cloud.get_all_endpoints()

        self.assertEqual(endpoints, mock_endpoints)
        mock_request.assert_called_once()

    @patch("vespa.deployment.VespaCloud.get_all_endpoints")
    def test_get_mtls_endpoint(self, mock_get_all_endpoints):
        mock_endpoints = [
            {"url": "https://endpoint1.vespa.oath.cloud:4443", "authMethod": "mtls"},
            {"url": "https://endpoint2.vespa.oath.cloud:4443", "authMethod": "token"},
        ]
        mock_get_all_endpoints.return_value = mock_endpoints

        mtls_endpoint = self.vespa_cloud.get_mtls_endpoint()

        self.assertEqual(mtls_endpoint, "https://endpoint1.vespa.oath.cloud:4443")

    @patch("vespa.deployment.VespaCloud._request")
    def test_get_private_services(self, mock_request):
        mock_response = {
            "privateServices": [
                {
                    "allowedUrns": [
                        {"type": "aws-private-link", "urn": "arn:aws:iam::123456:root"}
                    ],
                    "authMethods": ["mtls"],
                    "cluster": "default",
                    "endpoints": [],
                    "serviceId": "com.amazonaws.vpce.us-east-1.vpce-svc-abc123",
                    "type": "aws-private-link",
                }
            ]
        }
        mock_request.return_value = mock_response

        result = self.vespa_cloud.get_private_services()

        self.assertEqual(result, mock_response)
        mock_request.assert_called_once_with(
            "GET",
            "/application/v4/tenant/test_tenant/application/test_app/instance/default/environment/dev/region/dev-region/private-services",
        )

    @patch("vespa.deployment.VespaCloud._request")
    def test_get_app_package_contents(self, mock_request):
        mock_request.return_value = [
            "https://endpoint/content/README.md",
            "https://endpoint/content/schemas/music.sd",
        ]
        result = self.vespa_cloud.get_app_package_contents(
            "instance", region="region", environment="env"
        )
        self.assertIsInstance(result, list)
        self.assertIn("https://endpoint/content/README.md", result)

    @patch("vespa.deployment.VespaCloud._request")
    def test_get_schemas(self, mock_request):
        # First call returns the list of schema endpoints
        # Next calls return the content for each schema
        # _request returns strings (via response.text), not bytes
        mock_request.side_effect = [
            ["schemas/schema1.sd", "schemas/schema2.sd"],  # First call
            "schema1 content",  # Second call (for schema1)
            "schema2 content",  # Third call (for schema2)
        ]
        result = self.vespa_cloud.get_schemas("instance", "region", "env")
        self.assertIsInstance(result, dict)
        self.assertIn("schema1 content", result.values())
        self.assertIn("schema2 content", result.values())
        self.assertTrue(
            all(isinstance(schema_content, str) for schema_content in result.values())
        )
        self.assertTrue(
            all(isinstance(schema_name, str) for schema_name in result.keys())
        )

    @patch("vespa.deployment.VespaCloud.get_app_package_contents")
    @patch("vespa.deployment.VespaCloud._request")
    def test_download_app_package_content(self, mock_request, mock_get_app):
        mock_get_app.return_value = ["https://endpoint/content/README.md"]
        mock_request.return_value = b"file content"

        with TemporaryDirectory() as temp_dir:
            self.vespa_cloud.download_app_package_content(
                temp_dir, instance="instance", region="region", environment="env"
            )
            expected_file = os.path.join(temp_dir, "README.md")
            self.assertTrue(os.path.exists(expected_file))

    @patch("vespa.deployment.VespaCloud._request")
    def test_follow_deployment_success(self, mock_request):
        mock_request.side_effect = [
            {"active": True, "log": {}, "lastId": 1},
            {"active": True, "log": {}, "lastId": 2},
            {"active": False, "status": "success", "log": {}, "lastId": 3},
        ]

        self.vespa_cloud._follow_deployment("default", "dev-us-east-1", 123)

        self.assertEqual(mock_request.call_count, 3)

    @patch("vespa.deployment.VespaCloud._request")
    def test_follow_deployment_failure(self, mock_request):
        mock_request.side_effect = [
            {"active": True, "log": {}, "lastId": 1},
            {"active": False, "status": "error", "log": {}, "lastId": 2},
        ]

        with self.assertRaises(RuntimeError):
            self.vespa_cloud._follow_deployment("default", "dev-us-east-1", 123)

    @patch("vespa.deployment.VespaCloud._request")
    def test_check_production_build_status_deployed(self, mock_request):
        mock_request.return_value = {"deployed": True, "status": "done"}

        status = self.vespa_cloud.check_production_build_status(456)

        self.assertEqual(status, {"deployed": True, "status": "done"})
        mock_request.assert_called_once_with(
            "GET",
            "/application/v4/tenant/test_tenant/application/test_app/build-status/456",
        )

    @patch("vespa.deployment.VespaCloud._request")
    def test_check_production_build_status_deploying(self, mock_request):
        mock_request.return_value = {"deployed": False, "status": "deploying"}

        status = self.vespa_cloud.check_production_build_status(456)

        self.assertEqual(status, {"deployed": False, "status": "deploying"})

    @patch("vespa.deployment.VespaCloud._request")
    def test_wait_for_prod_deployment_raises_on_failed_job(self, mock_request):
        mock_request.return_value = {
            "deployed": False,
            "status": "deploying",
            "jobs": [
                {"jobName": "production-us-central-1", "runStatus": "success"},
                {"jobName": "production-us-east-3", "runStatus": "deploymentFailed"},
            ],
        }

        with self.assertRaises(RuntimeError) as ctx:
            self.vespa_cloud.wait_for_prod_deployment(456)
        self.assertIn("production-us-east-3: deploymentFailed", str(ctx.exception))

    @patch("vespa.deployment.VespaCloud._try_get_access_token")
    def test_try_get_access_token(self, mock_get_token):
        mock_get_token.return_value = "fake_access_token"

        token = self.vespa_cloud._try_get_access_token()

        self.assertEqual(token, "fake_access_token")

    @patch("vespa.deployment.VespaCloud._request")
    def test_get_endpoint_auth_method(self, mock_request):
        mock_request.return_value = {
            "endpoints": [
                {
                    "url": "https://endpoint1.vespa.oath.cloud:4443",
                    "authMethod": "mtls",
                },
                {
                    "url": "https://endpoint2.vespa.oath.cloud:4443",
                    "authMethod": "token",
                },
            ]
        }

        auth_method = self.vespa_cloud.get_endpoint_auth_method(
            "https://endpoint2.vespa.oath.cloud:4443"
        )

        self.assertEqual(auth_method, "token")

    @patch("vespa.deployment.VespaCloud.get_all_endpoints")
    def test_get_token_endpoint(self, mock_get_all_endpoints):
        mock_get_all_endpoints.return_value = [
            {"url": "https://endpoint1.vespa.oath.cloud:4443", "authMethod": "mtls"},
            {"url": "https://endpoint2.vespa.oath.cloud:4443", "authMethod": "token"},
        ]

        token_endpoint = self.vespa_cloud.get_token_endpoint()

        self.assertEqual(token_endpoint, "https://endpoint2.vespa.oath.cloud:4443")

    @patch("vespa.deployment.VespaCloud._load_certificate_pair")
    def test_load_certificate_pair(self, mock_load_cert):
        mock_private_key = MagicMock()
        mock_certificate = MagicMock()
        mock_load_cert.return_value = (mock_private_key, mock_certificate)

        private_key, certificate = self.vespa_cloud._load_certificate_pair()

        self.assertEqual(private_key, mock_private_key)
        self.assertEqual(certificate, mock_certificate)

    @patch("subprocess.run")
    def test_check_vespacli_available(self, mock_run):
        mock_run.return_value.stdout = b"Vespa CLI version 1.0.0"

        is_available = self.vespa_cloud._check_vespacli_available()

        self.assertTrue(is_available)

    @patch("subprocess.run")
    def test_check_vespacli_not_available(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        is_available = self.vespa_cloud._check_vespacli_available()

        self.assertFalse(is_available)

    @patch("vespa.deployment.VespaCloud._request")
    def test_get_last_deployable(self, mock_request):
        mock_request.return_value = {
            "builds": [
                {"build": 456, "deployable": True},
                {"build": 455, "deployable": False},
                {"build": 454, "deployable": True},
            ]
        }

        last_deployable = self.vespa_cloud._get_last_deployable(456)

        self.assertEqual(last_deployable, 456)

    @patch("vespa.deployment.VespaCloud._request")
    def test_get_last_deployable_no_deployable(self, mock_request):
        mock_request.return_value = {
            "builds": [
                {"build": 456, "deployable": False},
                {"build": 455, "deployable": False},
                {"build": 454, "deployable": False},
            ]
        }

        with self.assertRaises(Exception):
            self.vespa_cloud._get_last_deployable(456)


class TestVaultAccessRules(unittest.TestCase):
    def setUp(self):
        self.tenant = "test_tenant"
        self.application = "test_app"
        self.application_package = MagicMock()
        VespaCloud._try_get_access_token = MagicMock(return_value="fake_access_token")
        self.vespa_cloud = VespaCloud(
            tenant=self.tenant,
            application=self.application,
            application_package=self.application_package,
        )

    # --- XML parsing tests ---

    def test_parse_vault_names_no_secrets(self):
        xml = "<services><container></container></services>"
        result = VespaCloud._parse_vault_names_from_services_xml(xml)
        self.assertEqual(result, set())

    def test_parse_vault_names_single_vault(self):
        xml = """<services>
          <container>
            <secrets>
              <openAiToken vault="my-vault" name="openai-key"/>
            </secrets>
          </container>
        </services>"""
        result = VespaCloud._parse_vault_names_from_services_xml(xml)
        self.assertEqual(result, {"my-vault"})

    def test_parse_vault_names_multiple_vaults(self):
        xml = """<services>
          <container>
            <secrets>
              <openAiToken vault="vault-a" name="key-1"/>
              <cohereToken vault="vault-b" name="key-2"/>
            </secrets>
          </container>
        </services>"""
        result = VespaCloud._parse_vault_names_from_services_xml(xml)
        self.assertEqual(result, {"vault-a", "vault-b"})

    def test_parse_vault_names_multiple_secrets_blocks(self):
        xml = """<services>
          <container id="c1">
            <secrets>
              <token vault="vault-1" name="key-1"/>
            </secrets>
          </container>
          <container id="c2">
            <secrets>
              <token vault="vault-2" name="key-2"/>
            </secrets>
          </container>
        </services>"""
        result = VespaCloud._parse_vault_names_from_services_xml(xml)
        self.assertEqual(result, {"vault-1", "vault-2"})

    def test_parse_vault_names_malformed_xml(self):
        xml = "<services><not-closed>"
        result = VespaCloud._parse_vault_names_from_services_xml(xml)
        self.assertEqual(result, set())

    # --- Vault rule checking tests ---

    @patch("vespa.deployment.VespaCloud._request")
    def test_ensure_vault_access_rule_already_has_access(self, mock_request):
        mock_request.return_value = {
            "rules": [
                {
                    "id": 0,
                    "application": "test_app",
                    "contexts": [VespaCloud.secret_store_dev_alias],
                },
            ]
        }
        self.vespa_cloud._ensure_vault_access_rule("my-vault")
        # Should only GET, no PUT (no CSRF fetch either)
        mock_request.assert_called_once_with(
            "GET",
            "/tenant-secret/v1/tenant/test_tenant/vault/my-vault",
        )

    @patch("vespa.deployment.VespaCloud._request")
    def test_ensure_vault_access_rule_needs_access(self, mock_request):
        mock_request.side_effect = [
            # GET response: no rules for this app
            {
                "rules": [
                    {
                        "id": 0,
                        "application": "other_app",
                        "contexts": [VespaCloud.secret_store_dev_alias],
                    },
                ]
            },
            # GET CSRF token
            {"token": "fake-csrf-token"},
            # PUT response: confirms both rules
            {
                "rules": [
                    {
                        "id": 0,
                        "application": "other_app",
                        "contexts": [VespaCloud.secret_store_dev_alias],
                    },
                    {
                        "id": 1,
                        "application": "test_app",
                        "contexts": [VespaCloud.secret_store_dev_alias],
                    },
                ]
            },
        ]
        self.vespa_cloud._ensure_vault_access_rule("my-vault")
        self.assertEqual(mock_request.call_count, 3)
        # Verify CSRF fetch
        self.assertEqual(mock_request.call_args_list[1][0], ("GET", "/csrf/v1"))
        # Verify PUT call
        put_call = mock_request.call_args_list[2]
        self.assertEqual(put_call[0][0], "PUT")
        self.assertEqual(
            put_call[0][2]["rules"],
            [
                {
                    "id": 0,
                    "application": "other_app",
                    "contexts": [VespaCloud.secret_store_dev_alias],
                },
                {
                    "id": 1,
                    "application": "test_app",
                    "contexts": [VespaCloud.secret_store_dev_alias],
                },
            ],
        )
        self.assertEqual(put_call[0][3]["vespa-csrf-token"], "fake-csrf-token")

    @patch("vespa.deployment.VespaCloud._request")
    def test_ensure_vault_access_rule_empty_rules(self, mock_request):
        mock_request.side_effect = [
            {"rules": []},
            # GET CSRF token
            {"token": "fake-csrf-token"},
            # PUT response confirms the new rule
            {
                "rules": [
                    {
                        "id": 0,
                        "application": "test_app",
                        "contexts": [VespaCloud.secret_store_dev_alias],
                    }
                ]
            },
        ]
        self.vespa_cloud._ensure_vault_access_rule("my-vault")
        self.assertEqual(mock_request.call_count, 3)
        self.assertEqual(
            mock_request.call_args_list[2][0][2]["rules"],
            [
                {
                    "id": 0,
                    "application": "test_app",
                    "contexts": [VespaCloud.secret_store_dev_alias],
                }
            ],
        )

    @patch("vespa.deployment.VespaCloud._request")
    def test_ensure_vault_access_rule_put_response_missing_rule(self, mock_request):
        mock_request.side_effect = [
            {"rules": []},
            # GET CSRF token
            {"token": "fake-csrf-token"},
            # PUT response does not confirm the rule
            {"rules": []},
        ]
        with self.assertRaises(RuntimeError):
            self.vespa_cloud._ensure_vault_access_rule("my-vault")

    @patch("vespa.deployment.VespaCloud._request")
    def test_ensure_vault_access_rule_put_response_not_dict(self, mock_request):
        mock_request.side_effect = [
            {"rules": []},
            # GET CSRF token
            {"token": "fake-csrf-token"},
            "unexpected string response",
        ]
        with self.assertRaises(RuntimeError):
            self.vespa_cloud._ensure_vault_access_rule("my-vault")

    # --- Error handling tests ---

    @patch("vespa.deployment.logging")
    @patch("vespa.deployment.VespaCloud._request")
    def test_ensure_vault_access_for_dev_api_failure_warns(
        self, mock_request, mock_logging
    ):
        self.application_package.services_to_text = """<services>
          <container>
            <secrets>
              <token vault="failing-vault" name="key"/>
            </secrets>
          </container>
        </services>"""
        mock_request.side_effect = RuntimeError("API error")
        # Should not raise — just warn via logging
        self.vespa_cloud._ensure_vault_access_for_dev()
        mock_logging.warning.assert_called_once()
        warning_msg = mock_logging.warning.call_args[0][0]
        self.assertIn("Failed to set vault access rule", warning_msg)

    @patch("vespa.deployment.VespaCloud._request")
    def test_ensure_vault_access_rule_api_key_raises(self, mock_request):
        """When auth is api_key and the rule is missing, ValueError is raised."""
        mock_request.return_value = {"rules": []}
        self.vespa_cloud.control_plane_auth_method = "api_key"
        with self.assertRaises(ValueError) as ctx:
            self.vespa_cloud._ensure_vault_access_rule("my-vault")
        self.assertIn(
            "API key authentication does not have permission", str(ctx.exception)
        )
        self.assertIn("key_location", str(ctx.exception))
        # Should only GET, no PUT attempted
        mock_request.assert_called_once()

    @patch("vespa.deployment.logging")
    @patch("vespa.deployment.VespaCloud._request")
    def test_ensure_vault_access_for_dev_api_key_raises(
        self, mock_request, mock_logging
    ):
        """ValueError from api_key auth check propagates (not caught as warning)."""
        self.application_package.services_to_text = """<services>
          <container>
            <secrets>
              <token vault="my-vault" name="key"/>
            </secrets>
          </container>
        </services>"""
        mock_request.return_value = {"rules": []}
        self.vespa_cloud.control_plane_auth_method = "api_key"
        with self.assertRaises(ValueError) as ctx:
            self.vespa_cloud._ensure_vault_access_for_dev()
        self.assertIn(
            "API key authentication does not have permission", str(ctx.exception)
        )
        mock_logging.warning.assert_not_called()

    # --- Integration: _ensure_vault_access_for_dev ---

    @patch("vespa.deployment.VespaCloud._ensure_vault_access_rule")
    def test_ensure_vault_access_for_dev_calls_per_vault(self, mock_ensure_rule):
        self.application_package.services_to_text = """<services>
          <container>
            <secrets>
              <t1 vault="vault-a" name="key-1"/>
              <t2 vault="vault-b" name="key-2"/>
            </secrets>
          </container>
        </services>"""
        self.vespa_cloud._ensure_vault_access_for_dev()
        called_vaults = sorted(call[0][0] for call in mock_ensure_rule.call_args_list)
        self.assertEqual(called_vaults, ["vault-a", "vault-b"])

    @patch("vespa.deployment.VespaCloud._ensure_vault_access_rule")
    def test_ensure_vault_access_for_dev_no_secrets(self, mock_ensure_rule):
        self.application_package.services_to_text = (
            "<services><container></container></services>"
        )
        self.vespa_cloud._ensure_vault_access_for_dev()
        mock_ensure_rule.assert_not_called()

    # --- _get_services_xml_content tests ---

    def test_get_services_xml_content_from_package(self):
        self.application_package.services_to_text = "<services/>"
        result = self.vespa_cloud._get_services_xml_content()
        self.assertEqual(result, "<services/>")

    def test_get_services_xml_content_from_disk(self):
        self.vespa_cloud.application_package = None
        with TemporaryDirectory() as tmp:
            services_path = os.path.join(tmp, "services.xml")
            with open(services_path, "w") as f:
                f.write("<services><container/></services>")
            self.vespa_cloud.application_root = tmp
            result = self.vespa_cloud._get_services_xml_content()
            self.assertEqual(result, "<services><container/></services>")

    def test_get_services_xml_content_none(self):
        self.vespa_cloud.application_package = None
        self.vespa_cloud.application_root = None
        result = self.vespa_cloud._get_services_xml_content()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
