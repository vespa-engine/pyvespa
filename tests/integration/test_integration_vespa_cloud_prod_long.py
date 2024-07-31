import pathlib
import os
from datetime import datetime, timedelta
import unittest

from vespa.deployment import VespaCloud
from vespa.package import (
    ApplicationPackage,
    EmptyDeploymentConfiguration,
    Validation,
    ValidationID,
)


class TestDeployProdWithTests(unittest.TestCase):
    def setUp(self) -> None:
        # Set root to parent directory/testapps/production-deployment-with-tests
        self.application_root = (
            pathlib.Path(__file__).parent.parent
            / "testapps"
            / "production-deployment-with-tests"
        )
        self.vespa_cloud = VespaCloud(
            tenant="vespa-team",
            application="pyvespa-integration",
            key_content=os.getenv("VESPA_TEAM_API_KEY").replace(r"\n", "\n"),
            application_root=self.application_root,
        )

        self.build_no = self.vespa_cloud.deploy_to_prod(
            source_url="https://github.com/vespa-engine/pyvespa",
        )

    def test_application_status(self):
        # Wait for deployment to be ready
        success = self.vespa_cloud.wait_for_prod_deployment(
            build_no=self.build_no, max_wait=3600 * 4
        )
        if not success:
            self.fail("Deployment failed")
        self.app = self.vespa_cloud.get_application(environment="prod")

    def tearDown(self) -> None:
        # Deployment is deleted by deploying with an empty deployment.xml file
        # Creating a dummy ApplicationPackage just to use the validation_to_text method
        app_package = ApplicationPackage(name="empty")
        # Vespa won't push the deleted deployment.xml file unless we add a validation override
        tomorrow = datetime.now() + timedelta(days=1)
        formatted_date = tomorrow.strftime("%Y-%m-%d")
        app_package.validations = [
            Validation(ValidationID("deployment-removal"), formatted_date)
        ]
        # Write validations_to_text to "validation-overrides.xml"
        with open(self.application_root / "validation-overrides.xml", "w") as f:
            f.write(app_package.validations_to_text)
        # Create an empty deployment.xml file
        app_package.deployment_config = EmptyDeploymentConfiguration()
        with open(self.application_root / "deployment.xml", "w") as f:
            f.write(app_package.deployment_config.to_xml_string())
        # This will delete the deployment
        self.vespa_cloud._start_prod_deployment(self.application_root)
