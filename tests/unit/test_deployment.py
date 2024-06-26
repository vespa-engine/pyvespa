import unittest
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


if __name__ == "__main__":
    unittest.main()
