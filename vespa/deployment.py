import httpx
from urllib3.exceptions import HTTPError
import json
import os
import sys
import zipfile
import logging
from base64 import standard_b64encode
from datetime import datetime
from io import BytesIO
from pathlib import Path
from time import sleep, strftime, gmtime
from typing import Tuple, Union, IO, Optional, List, Dict, Literal
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import timezone
import platform
import subprocess
import shlex
import select
from dateutil import parser
import time
from urllib.parse import urlparse

import docker
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec

from vespa.application import Vespa
from vespa.package import ApplicationPackage
import vespa

# Get the Vespa home directory
VESPA_HOME = Path(os.getenv("VESPA_HOME", Path.home() / ".vespa"))


class VespaDeployment:
    def read_app_package_from_disk(self, application_root: Path) -> bytes:
        """
        Reads the contents of an application package on disk into a zip file.

        Args:
            application_root (str): The directory root of the application package.

        Returns:
            bytes: The zipped application package.
        """

        tmp_zip = "tmp_app_package.zip"
        orig_dir = os.getcwd()
        zipf = zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED)
        os.chdir(application_root)  # Workaround to avoid the top-level directory
        for root, dirs, files in os.walk(".", followlinks=True):
            for file in files:
                zipf.write(os.path.join(root, file))
        zipf.close()
        os.chdir(orig_dir)
        with open(tmp_zip, "rb") as f:
            data = f.read()
        os.remove(tmp_zip)

        return data


class VespaDocker(VespaDeployment):
    def __init__(
        self,
        url: str = "http://localhost",
        port: int = 8080,
        container_memory: Union[str, int] = 4 * (1024**3),
        output_file: IO = sys.stdout,
        container: Optional[docker.models.containers.Container] = None,
        container_image: str = "vespaengine/vespa",
        volumes: Optional[List[str]] = None,
        cfgsrv_port: int = 19071,
        debug_port: int = 5005,
    ) -> None:
        """
        Manage Docker deployments.

        Make sure to start the Docker daemon before instantiating this class.

        Example usage:
            ```python
            from vespa.deployment import VespaDocker

            vespa_docker = VespaDocker(port=8080)
            # or initialize from a running container:
            vespa_docker = VespaDocker('http://localhost', 8080, None, None, 4294967296, 'vespaengine/vespa')
            ```

        **Note**:

        It is **NOT** possible to refer to Volume Mounts in your Application Package.
        This means that for example .onnx-model files that are part of the Application Package **must** be on your host machine, so
        that it can be uploaded as part of the Application Package to the Vespa container.

        Args:
            port (int): The port for the container. Default is 8080.
            cfgsrv_port (int): The Vespa Config Server port. Default is 19071.
            debug_port (int): The port to connect to for debugging the Vespa container. Default is 5005.
            output_file (str): The file to write output messages to.
            container_memory (int): Memory available to the container in bytes. Default is 4GB.
            container (str, optional): Used when instantiating `VespaDocker` from a running container.
            volumes (list of str, optional): A list of volume mount strings, such as `['/home/user1/:/mnt/vol2', '/var/www:/mnt/vol1']`. The Application Package cannot reference volume mounts.
            container_image (str): The Docker container image to use.
            url (str): The URL to connect to the Vespa instance. Default is "http://localhost".

        """

        self.container = container
        container_id = None
        container_name = None
        if container:
            container_id = container.id
            container_name = container.name
        self.container_name = container_name
        self.container_id = container_id
        self.url = url
        self.local_port = port
        self.cfgsrv_port = cfgsrv_port
        self.debug_port = debug_port
        self.container_memory = container_memory
        self.volumes = volumes
        self.output = output_file
        self.container_image = container_image

        if os.getenv("PYVESPA_DEBUG") == "true":
            logging.basicConfig(level=logging.DEBUG)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.container_id == other.container_id
            and self.container_name == other.container_name
            and self.url == other.url
            and self.local_port == other.local_port
            and self.container_memory == other.container_memory
            and self.container_image.split(":")[0]
            == other.container_image.split(":")[0]
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3}, {4}, {5}, {6})".format(
            self.__class__.__name__,
            repr(self.url),
            repr(self.local_port),
            repr(self.container_name),
            repr(self.container_id),
            repr(self.container_memory),
            repr(self.container_image.split(":")[0]),
        )

    @staticmethod
    def from_container_name_or_id(
        name_or_id: str, output_file: IO = sys.stdout
    ) -> "VespaDocker":
        """
        Instantiate VespaDocker from a running container.

        Args:
            name_or_id (str): The name or id of the running container.
            output_file (str): The file to write output messages to.

        Raises:
            ValueError: If the specified container is not found.

        Returns:
            VespaDocker: An instance of VespaDocker associated with the running container.
        """

        client = docker.from_env()
        try:
            container = client.containers.get(name_or_id)
        except docker.errors.NotFound:
            raise ValueError("The container does not exist.")
        port = int(
            container.attrs["HostConfig"]["PortBindings"]["8080/tcp"][0]["HostPort"]
        )
        container_memory = container.attrs["HostConfig"]["Memory"]
        container_image = container.image.tags[0]  # vespaengine/vespa:latest
        container_image_split = container_image.split("/")
        if len(container_image_split) > 2:
            # Means registry is included, e.g. docker.io/vespaengine/vespa:latest
            # This causes equality test to fail, so we remove the registry part
            container_image = "/".join(container_image_split[-2:])
        return VespaDocker(
            port=port,
            container_memory=container_memory,
            output_file=output_file,
            container=container,
            container_image=container_image,
        )

    def deploy(
        self,
        application_package: ApplicationPackage,
        max_wait_configserver: int = 60,
        max_wait_deployment: int = 300,
        max_wait_docker: int = 300,
        debug: bool = False,
    ) -> Vespa:
        """
        Deploy the application package into a Vespa container.

        Args:
            application_package (ApplicationPackage): The application package to be deployed.
            max_wait_configserver (int): Maximum seconds to wait for the config server to start.
            max_wait_deployment (int): Maximum seconds to wait for the deployment to complete.
            max_wait_docker (int): Maximum seconds to wait for the Docker container to start.
            debug (bool): If True, adds the configured debug_port to the Docker port mapping.

        Returns:
            VespaConnection: A Vespa connection instance once the deployment is complete.
        """
        return self._deploy_data(
            application_package,
            application_package.to_zip(),
            max_wait_configserver=max_wait_configserver,
            max_wait_application=max_wait_deployment,
            docker_timeout=max_wait_docker,
            debug=debug,
        )

    def deploy_from_disk(
        self,
        application_name: str,
        application_root: Path,
        max_wait_configserver: int = 60,
        max_wait_application: int = 300,
        docker_timeout: int = 300,
        debug: bool = False,
    ) -> Vespa:
        """
        Deploy from a directory tree.

        This method is used when making changes to application package files that are not supported by pyvespa.
        This is why this method is not found in the ApplicationPackage class.

        Args:
            application_name (str): The name of the application package.
            application_root (str): The root directory of the application package.
            debug (bool): If True, adds the configured debug_port to the Docker port mapping.

        Returns:
            VespaConnection: A Vespa connection instance once the deployment is complete.
        """

        data = self.read_app_package_from_disk(application_root)
        return self._deploy_data(
            ApplicationPackage(name=application_name),
            data,
            debug,
            max_wait_application=max_wait_application,
            max_wait_configserver=max_wait_configserver,
            docker_timeout=docker_timeout,
        )

    def wait_for_config_server_start(self, max_wait: int = 300) -> None:
        """
        Waits for the Config Server to start inside the Docker image.

        Args:
            max_wait (int): The maximum number of seconds to wait for the application endpoint to become available.

        Raises:
            RuntimeError: If the config server does not start within the specified max_wait time.

        Returns:
            None
        """

        try_interval = 5
        waited = 0
        while not self._check_configuration_server() and (waited < max_wait):
            print(
                "Waiting for configuration server, {0}/{1} seconds...".format(
                    waited, max_wait
                ),
                file=self.output,
            )
            sleep(try_interval)
            waited += try_interval
        if waited >= max_wait:
            self.dump_vespa_log()
            raise RuntimeError(
                "Config server did not start, waited for {0} seconds.".format(max_wait)
            )

    def start_services(self, max_wait: int = 120) -> None:
        """
        Start Vespa services inside the Docker image, first waiting for the Config Server, then for other services.

        Args:
            max_wait (int): The maximum number of seconds to wait for the application endpoint to become available.

        Raises:
            RuntimeError: If a container has not been set or the services fail to start within the specified max_wait time.

        Returns:
            None
        """

        if self.container:
            start_config = self.container.exec_run(
                "bash -c '/opt/vespa/bin/vespa-start-configserver'"
            )
            while not self._check_configuration_server():
                print("Waiting for configuration server.", file=self.output)
                sleep(5)
            for line in start_config.output.decode("utf-8").split("\n"):
                print(line, file=self.output)
            start_services = self.container.exec_run(
                "bash -c '/opt/vespa/bin/vespa-start-services'"
            )
            app = Vespa(
                url=self.url,
                port=self.local_port,
            )
            app.wait_for_application_up(max_wait=max_wait)
            for line in start_services.output.decode("utf-8").split("\n"):
                print(line, file=self.output)
        else:
            raise RuntimeError("No container found")

    def stop_services(self) -> None:
        """
        Stop Vespa services inside the Docker image, first stopping the services, then stopping the Config Server.

        Raises:
            RuntimeError: If a container has not been set or an error occurs while stopping the services.

        Returns:
            None
        """

        if self.container:
            stop_services = self.container.exec_run(
                "bash -c '/opt/vespa/bin/vespa-stop-services'"
            )
            for line in stop_services.output.decode("utf-8").split("\n"):
                print(line, file=self.output)
            stop_config = self.container.exec_run(
                "bash -c '/opt/vespa/bin/vespa-stop-configserver'"
            )
            for line in stop_config.output.decode("utf-8").split("\n"):
                print(line, file=self.output)
        else:
            raise RuntimeError("No container found")

    def restart_services(self) -> None:
        """
        Restart Vespa services inside the Docker image. This is equivalent to calling
        `self.stop_services()` followed by `self.start_services()`.

        Raises:
            RuntimeError: If a container has not been set or an error occurs during the restart process.

        Returns:
            None
        """
        self.stop_services()
        self.start_services()

    def dump_vespa_log(self) -> None:
        log_dump = self.container.exec_run(
            "bash -c 'cat /opt/vespa/logs/vespa/vespa.log'"
        )
        logging.debug("Dumping vespa.log:")
        logging.debug(log_dump.output.decode("utf-8"))

    def _deploy_data(
        self,
        application: ApplicationPackage,
        data,
        debug: bool,
        max_wait_configserver: int,
        max_wait_application: int,
        docker_timeout: int,
    ) -> Vespa:
        """
        Deploys an Application Package as zipped data.

        Args:
            application (ApplicationPackage): The application package to be deployed.
            max_wait_configserver (int): Seconds to wait for the config server to start.
            max_wait_application (int): Seconds to wait for the application deployment.

        Raises:
            RuntimeError: If the deployment fails or if there is an issue with the application package.

        Returns:
            Vespa: A Vespa connection instance after the deployment is successful.
        """
        self._run_vespa_engine_container(
            application_name=application.name,
            container_memory=self.container_memory,
            volumes=self.volumes,
            debug=debug,
            docker_timeout=docker_timeout,
        )
        self.wait_for_config_server_start(max_wait=max_wait_configserver)

        r = requests.post(
            "http://localhost:{}/application/v2/tenant/default/prepareandactivate".format(
                self.cfgsrv_port
            ),
            headers={"Content-Type": "application/zip"},
            data=data,
            verify=False,
        )
        logging.debug("Deploy status code: {}".format(r.status_code))
        if r.status_code != 200:
            raise RuntimeError(
                "Deployment failed, code: {}, message: {}".format(
                    r.status_code, json.loads(r.content.decode("utf8"))
                )
            )

        app = Vespa(url=self.url, port=self.local_port, application_package=application)
        app.wait_for_application_up(max_wait=max_wait_application)

        print("Finished deployment.", file=self.output)
        return app

    def _run_vespa_engine_container(
        self,
        application_name: str,
        container_memory: str,
        volumes: List[str],
        debug: bool,
        docker_timeout: int,
    ) -> None:
        client = docker.from_env(timeout=docker_timeout)
        if self.container is None:
            try:
                logging.debug("Try Docker container restart")
                self.container = client.containers.get(application_name)
                self.container.restart()
            except docker.errors.NotFound:
                mapped_ports = {8080: self.local_port, 19071: self.cfgsrv_port}
                if debug:
                    mapped_ports[self.debug_port] = self.debug_port
                logging.debug(
                    "Start a Docker container: "
                    "image: {image}, "
                    "mem_limit: {mem_limit}, "
                    "name: {name}, "
                    "hostname: {hostname}, "
                    "ports: {ports}, "
                    "volumes: {volumes}".format(
                        image=self.container_image,
                        mem_limit=container_memory,
                        name=application_name,
                        hostname=application_name,
                        ports=mapped_ports,
                        volumes=volumes,
                    )
                )
                self.container = client.containers.run(
                    self.container_image,
                    detach=True,
                    mem_limit=container_memory,
                    name=application_name,
                    hostname=application_name,
                    privileged=True,
                    ports=mapped_ports,
                    volumes=volumes,
                )
            self.container_name = self.container.name
            self.container_id = self.container.id
        else:
            logging.debug("Try Docker container restart")
            self.container.restart()

    def _check_configuration_server(self) -> bool:
        """
        Check if the configuration server is running and ready for deployment.

        Returns:
            bool: True if the configuration server is running and ready for deployment, False otherwise.
        """
        if self.container is None:
            return False

        output = self.container.exec_run(
            "bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'"
        ).output.decode("utf-8")
        logging.debug("Config Server ApplicationStatus head response: " + output)
        return output.split("\r\n")[0] == "HTTP/1.1 200 OK"


class VespaCloud(VespaDeployment):
    def __init__(
        self,
        tenant: str,
        application: str,
        application_package: Optional[ApplicationPackage] = None,
        key_location: Optional[str] = None,
        key_content: Optional[str] = None,
        auth_client_token_id: Optional[str] = None,
        output_file: IO = sys.stdout,
        application_root: Optional[str] = None,
        cluster: Optional[str] = None,
        instance: str = "default",
    ) -> None:
        """
        Deploy an application to the Vespa Cloud (cloud.vespa.ai).

        There are several ways to initialize VespaCloud:
        - Application source: From a Python-defined application package or from the application_root folder.
        - Control plane access: Using an API key (must be added to Vespa Cloud Console) or an access token, obtained by interactive login.
        - Data plane access: mTLS is used by default, but Vespa applications can also be configured to use token-based authentication (token must be added to Vespa Cloud Console, and the corresponding auth_token_id must be provided).

        Example usage:
            ```python
            # 1. Initialize VespaCloud with an application package and existing API key for control plane access.
            vespa_cloud = VespaCloud(
                tenant="my-tenant",
                application="my-application",
                application_package=app_package,
                key_location="/path/to/private-key.pem",
            )

            # 2. Initialize VespaCloud from disk folder by interactive control plane auth.
            vespa_cloud = VespaCloud(
                tenant="my-tenant",
                application="my-application",
                application_root="/path/to/application",
            )

            # 3. Initialize VespaCloud with an application package and token-based data plane access.
            vespa_cloud = VespaCloud(
                tenant="my-tenant",
                application="my-application",
                application_package=app_package,
                auth_client_token_id="my-token-id", # Must be added in Vespa Cloud Console
            )
            ```

        Args:
            tenant (str): Tenant name registered in the Vespa Cloud.
            application (str): Application name in the Vespa Cloud.
            application_package (ApplicationPackage): Application package to be deployed. Either this or application_root must be set.
            key_location (str, optional): Location of the control plane key used for signing HTTP requests to the Vespa Cloud.
            key_content (str, optional): Content of the control plane key used for signing HTTP requests to the Vespa Cloud. Use only when the key file is not available.
            auth_client_token_id (str, optional): Token-based data plane authentication. This token name must be configured in the Vespa Cloud Console. It configures Vespa's services.xml, and the token must have read and write permissions.
            output_file (str, optional): Output file to write output messages. Default is sys.stdout.
            application_root (str, optional): Directory for the application root (location of services.xml, models/, schemas/, etc.). If the application is packaged with Maven, use the generated `<myapp>/target/application` directory.
            cluster (str, optional): Name of the cluster to target when retrieving endpoints. This affects which endpoints are used for initializing the :class:`Vespa` instance in `VespaCloud.get_application` and `VespaCloud.deploy`.
            instance (str, optional): Name of the application instance. Default is "default".

        Raises:
            RuntimeError: If deployment fails.

        Returns:
            Vespa: A Vespa connection instance for interacting with the deployed application.
        """
        self.tenant = tenant
        self.application = application
        self.application_package = application_package
        self.application_root = application_root
        if self.application_package is None and self.application_root is None:
            raise ValueError(
                "Either application_package or application_root must be set for deployment."
            )
        self.instance = instance
        self.output = output_file
        self.api_key = self._read_private_key(key_location, key_content)
        self.control_plane_auth_method = None  # "api_key" or "access_token"
        self.control_plane_access_token = None
        self.auth_file_path = VESPA_HOME / "auth.json"
        if self._check_vespacli_available():
            # Run vespa config set application
            print("Setting application...")
            self._set_application()
            # Run vespa config set target cloud
            print("Setting target cloud...")
            self._set_target_cloud()
        if self.api_key:
            self.api_public_key_bytes = standard_b64encode(
                self.api_key.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            self.control_plane_auth_method = "api_key"
            print(
                "Api-key found for control plane access. Using api-key.",
                file=self.output,
            )
        else:
            self.api_public_key_bytes = None
            self.control_plane_auth_method = "access_token"
            print(
                "No api-key found for control plane access. Using access token.",
                file=self.output,
            )
            self.control_plane_access_token = self._try_get_access_token()
            print(
                "Successfully obtained access token for control plane access.",
                file=self.output,
            )
        self.data_cert_path = None
        self.data_key_path = None
        self.data_key, self.data_certificate = self._load_certificate_pair()
        self.default_timeout = (
            15  # seconds, default in httpx is 5. Eg. deployment may take longer.
        )
        self.httpx_limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=10,
        )
        self.base_url = "https://api-ctl.vespa-cloud.com:4443"
        self.pyvespa_version = vespa.__version__
        self.base_headers = {"User-Agent": f"pyvespa/{self.pyvespa_version}"}
        self.auth_client_token_id = auth_client_token_id
        self.cluster = cluster
        if auth_client_token_id is not None:
            if self.application_package is not None:
                # TODO: Should add some check to see if the auth_client_token_id is added to AuthClients.
                print(
                    "Auth client token id set. Make sure that corresponding auth_client is configured and added to ApplicationPackage.",
                    file=self.output,
                )
            else:
                print(
                    "Auth client token id set, but no application package provided. Make sure that services.xml is configured to use the provided token_id.",
                    file=self.output,
                )
        self.build_no = None  # Build number of submitted production deployment
        self.submitted_timestamp = None  # Timestamp of submitted production deployment

    def __enter__(self) -> "VespaCloud":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # Add property with getter and setter for self.build_no
    @property
    def build_no(self) -> Optional[int]:
        return self._build_no

    @build_no.setter
    def build_no(self, value: Optional[int]) -> None:
        self._build_no = value

    # Add property with getter and setter for  self.submitted_timestamp
    @property
    def submitted_timestamp(self) -> Optional[int]:
        return self._submitted_timestamp

    @submitted_timestamp.setter
    def submitted_timestamp(self, value: Optional[int]) -> None:
        self._submitted_timestamp = value

    def deploy(
        self,
        instance: Optional[str] = "default",
        disk_folder: Optional[str] = None,
        version: Optional[str] = None,
        max_wait: int = 1800,
        environment: Literal["dev", "perf"] = "dev",
    ) -> Vespa:
        """
        Deploy the given application package as the given instance in the Vespa Cloud dev or perf environment.

        Args:
            instance (str): Name of this instance of the application in the Vespa Cloud.
            disk_folder (str, optional): Disk folder to save the required Vespa config files. Defaults to the application name folder within the user's current working directory.
            version (str, optional): Vespa version to use for deployment. Defaults to None, meaning the latest version. Should only be set based on instructions from the Vespa team. Must be a valid Vespa version, e.g., "8.435.13".
            max_wait (int, optional): Seconds to wait for the deployment to complete.
            environment (Literal["dev", "perf"]): Environment to deploy to. Default is "dev".

        Returns:
            Vespa: A Vespa connection instance. This instance connects to the mTLS endpoint. To connect to the token endpoint, use `VespaCloud.get_application(endpoint_type="token")`.

        Raises:
            RuntimeError: If deployment fails or if there are issues with the deployment process.
        """
        if environment not in ["dev", "perf"]:
            raise ValueError(
                f"Invalid environment: {environment}. Must be 'dev' or 'perf'."
            )
        if self.application_package is not None:
            if disk_folder is None:
                disk_folder = os.path.join(os.getcwd(), self.application)
            Path(disk_folder).mkdir(parents=True, exist_ok=True)
            application_zip_bytes = self._to_application_zip(disk_folder=disk_folder)
        elif self.application_root is not None:
            disk_folder = self.application_root
            Path(disk_folder).mkdir(parents=True, exist_ok=True)
            client_pem_path = os.path.join(
                self.application_root, "security/clients.pem"
            )
            self._write_certificate(cert_path=client_pem_path)
            application_zip_bytes = BytesIO(
                self.read_app_package_from_disk(Path(disk_folder))
            )
        if environment == "perf":
            region = self.get_perf_region()
            job = "perf-" + region
        else:
            region = self.get_dev_region()
            job = "dev-" + region
        run = self._start_deployment(
            instance=instance or self.instance,
            job=job,
            disk_folder=disk_folder,
            application_zip_bytes=application_zip_bytes,
            version=version,
        )
        self._follow_deployment(instance, job, run, max_wait)
        app: Vespa = self.get_application(
            instance=instance, environment=environment, endpoint_type="mtls"
        )
        return app

    def deploy_to_prod(
        self,
        instance: Optional[str] = "default",
        application_root: Optional[str] = None,
        source_url: str = "",
    ) -> None:
        """
        Deploy the given application package as the given instance in the Vespa Cloud prod environment.
        NB! This feature is experimental and may fail in unexpected ways. Expect better support in future releases.

        If submitting an application that is not yet packaged, tests should be located in <application_root>/tests.
        If submitting an application packaged with maven, application_root should refer to the generated <myapp>/target/application directory.

        Args:
            instance (str): Name of this instance of the application in the Vespa Cloud.
            application_root (str): Path to either save the required Vespa config files (if initialized with application_package) or read them from (if initialized with application_root).
            source_url (str, optional): Optional source URL (including commit hash) for the deployment. This is a URL to the source code repository, e.g., GitHub, that is used to build the application package. Example: <https://github.com/vespa-cloud/vector-search/commit/474d7771bd938d35dc5dcfd407c21c019d15df3c>. The source URL will show up in the Vespa Cloud Console next to the build number.

        Raises:
            RuntimeError: If deployment fails or if there are issues with the deployment process.

        """
        if application_root is None:
            if self.application_root is None:
                application_root = os.path.join(os.getcwd(), self.application)
            else:
                application_root = self.application_root
        if self.application_package is not None:
            if self.application_package.deployment_config is None:
                raise ValueError("Prod deployment requires a deployment_config.")
            self.application_package.to_files(application_root)

        self.build_no = self._start_prod_deployment(
            application_root, source_url, instance
        )

        deploy_url = "https://console.vespa-cloud.com/tenant/{}/application/{}/prod/deployment".format(
            self.tenant, self.application
        )
        print(f"Follow deployment at: {deploy_url}", file=self.output)
        return self.build_no

    def _get_last_deployable(self, build_no: int) -> int:
        # This is due to optimization that some builds will not be deployable (e.g if no diff from previous build)
        # May take a few seconds for the build to show up in the deployment list
        max_wait = 5
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(1)
            deployments = self._request(
                "GET",
                f"/application/v4/tenant/{self.tenant}/application/{self.application}/deployment/",
            )
            if "builds" in deployments:
                builds = deployments["builds"]
                # Sort descending by build number
                sorted_builds = sorted(builds, key=lambda x: x["build"], reverse=True)
                for build in sorted_builds:
                    if build["build"] > build_no:
                        continue
                    if build["deployable"]:
                        return build["build"]
        raise Exception(
            "No deployable builds found within the time limit of 10 seconds."
        )

    def get_application(
        self,
        instance: str = "default",
        environment: str = "dev",
        endpoint_type: str = "mtls",
        vespa_cloud_secret_token: Optional[str] = None,
        region: Optional[str] = None,
        max_wait: int = 60,
    ) -> Vespa:
        """
        Get a connection to the Vespa application instance.
        Will only work if the application is already deployed.

        Example usage:
            ```python
            vespa_cloud = VespaCloud(...)
            app: Vespa = vespa_cloud.get_application()
            # Feed, query, visit, etc.
            ```

        Args:
            instance (str, optional): Name of this instance of the application in the Vespa Cloud. Default is "default".
            environment (str, optional): Environment of the application. Default is "dev". Options are "dev", "perf", or "prod".
            endpoint_type (str, optional): Type of endpoint to connect to. Default is "mtls". Options are "mtls" or "token".
            vespa_cloud_secret_token (str, optional): Vespa Cloud Secret Token. Only required if endpoint_type is "token".
            region (str, optional): Region of the application in Vespa Cloud, e.g., "aws-us-east-1c". If not provided, the first region from the environment will be used.
            max_wait (int, optional): Seconds to wait for the application to be up. Default is 60 seconds.

        Returns:
            Vespa: Vespa application instance.

        Raises:
            RuntimeError: If the application is not yet deployed or there are issues retrieving the connection.
        """

        if endpoint_type not in ["mtls", "token"]:
            raise ValueError("Endpoint type must be 'mtls' or 'token'.")
        if environment == "dev":
            region = self.get_dev_region()
            print(
                f"Only region: {region} available in dev environment.", file=self.output
            )
        elif environment == "perf":
            region = self.get_perf_region()
            print(
                f"Only region: {region} available in perf environment.",
                file=self.output,
            )
        elif environment == "prod":
            valid_regions = self.get_prod_regions(instance=instance or self.instance)
            if region is not None:
                if region not in valid_regions:
                    raise ValueError(
                        f"Region {region} not found in production regions: {valid_regions}"
                    )
            else:
                region = valid_regions[0]
        else:
            raise ValueError("Environment must be 'dev', 'perf', or 'prod'.")
        if endpoint_type == "mtls":
            mtls_endpoint = self.get_mtls_endpoint(
                instance=instance or self.instance,
                region=region,
                environment=environment,
            )
            app: Vespa = Vespa(
                url=mtls_endpoint,
                cert=self.data_cert_path,
                key=self.data_key_path or None,
                application_package=self.application_package,
            )
        elif endpoint_type == "token":
            try:  # May have client_token_id set but the deployed app was not configured to use it
                token_endpoint = self.get_token_endpoint(
                    instance=instance or self.instance,
                    region=region,
                    environment=environment,
                )
            except Exception as _:
                raise ValueError(
                    "No token endpoint found. Make sure the application is configured with a token endpoint."
                )
            if vespa_cloud_secret_token is None:
                raise ValueError(
                    "Vespa Cloud Secret Token must be provided for token based authentication."
                )
            app: Vespa = Vespa(
                url=token_endpoint,
                application_package=self.application_package,
                vespa_cloud_secret_token=vespa_cloud_secret_token,
            )
        app.wait_for_application_up(max_wait=max_wait)
        return app

    def check_production_build_status(self, build_no: Optional[int]) -> dict:
        """
        Check the status of a production build.
        Useful for example in CI/CD pipelines to check when a build has converged.

        Example usage:
            ```python
            vespa_cloud = VespaCloud(...)
            build_no = vespa_cloud.deploy_to_prod()
            status = vespa_cloud.check_production_build_status(build_no)
            # This can yield one of three responses:
            # 1. If the revision (build_no), or higher, has successfully converged everywhere, and nothing older has then been deployed on top of that again. Nothing more will happen in this case.
            # {
            #     "deployed": True,
            #     "status": "done"
            # }

            # 2. If the revision (build_no), or newer, has not yet converged, but the system is (most likely) still trying to deploy it. There is a point in polling again later when this is the response.
            # {
            #     "deployed": False,
            #     "status": "deploying"
            # }
            # 3. If the revision, or newer, has not yet converged everywhere, and it's never going to, because it was similar to the previous build, or marked obsolete by a user. There is no point in asking again for this revision.
            # {
            #     "deployed": False,
            #     "status": "done"
            # }
            ```

        Args:
            build_no (int): The build number to check.

        Returns:
            dict: A dictionary with the aggregated status of all deployment jobs for the given build number. The dictionary contains:
                - "deployed" (bool): Whether the build has successfully converged.
                - "status" (str): The current status of the build ("done", "deploying").

        Raises:
            RuntimeError: If there are issues with retrieving the status of the build.
        """

        if build_no is None:
            if self.build_no is None:
                raise ValueError("No build number provided, and no build number set.")
            else:
                build_no = int(self.build_no)
        print(f"Checking status of build number: {build_no}", file=self.output)
        status = self._request(
            "GET",
            f"/application/v4/tenant/{self.tenant}/application/{self.application}/build-status/{build_no}",
        )
        return status

    def wait_for_prod_deployment(
        self,
        build_no: Optional[int] = None,
        max_wait: int = 3600,
        poll_interval: int = 5,
    ) -> bool:
        """
        Wait for a production deployment to finish.
        Useful for example in CI/CD pipelines to wait for a deployment to finish.

        Example usage:
            ```python
            vespa_cloud = VespaCloud(...)
            build_no = vespa_cloud.deploy_to_prod()
            success = vespa_cloud.wait_for_prod_deployment(build_no, max_wait=3600, poll_interval=5)
            print(success)
            # Output: True
            ```

        Args:
            build_no (int): The build number to check.
            max_wait (int, optional): Maximum time to wait for the deployment in seconds. Default is 3600 (1 hour).
            poll_interval (int, optional): Polling interval in seconds. Default is 5 seconds.

        Returns:
            bool: True if the deployment is done and converged, False if the deployment has failed.

        Raises:
            TimeoutError: If the deployment did not finish within `max_wait` seconds.
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = self.check_production_build_status(build_no)
            if status["status"] == "done":
                return status["deployed"]
            time.sleep(poll_interval)
        raise TimeoutError(f"Deployment did not finish within {max_wait} seconds. ")

    def deploy_from_disk(
        self,
        instance: str,
        application_root: Path,
        max_wait: int = 300,
        version: Optional[str] = None,
        environment: str = "dev",
    ) -> Vespa:
        """
        Deploy to the development or performance environment from a directory tree.
        This method is used when making changes to application package files that are not supported by pyvespa.
        Note: Requires a certificate and key to be generated using 'vespa auth cert'.

        Example usage:
            ```python
            vespa_cloud = VespaCloud(...)
            vespa_cloud.deploy_from_disk(
                instance="my-instance",
                application_root="/path/to/application",
                max_wait=3600,
                version="8.435.13"
            )
            ```


        Args:
            instance (str): The name of the instance where the application will be run.
            application_root (str): The root directory of the application package.
            max_wait (int, optional): The maximum number of seconds to wait for the deployment. Default is 3600 (1 hour).
            version (str, optional): The Vespa version to use for the deployment. Default is None, which means the latest version. It must be a valid Vespa version (e.g., "8.435.13").
            environment (str, optional): Environment to deploy to. Default is "dev". Options are "dev" or "perf".

        Returns:
            Vespa: A Vespa connection instance. This connects to the mtls endpoint. To connect to the token endpoint, use `VespaCloud.get_application(endpoint_type="token")`.
        """

        data = BytesIO(self.read_app_package_from_disk(application_root))

        # Deploy the zipped application package
        disk_folder = os.path.join(os.getcwd(), self.application)
        if environment == "perf":
            region = self.get_perf_region()
            job = "perf-" + region
        elif environment == "dev":
            region = self.get_dev_region()
            job = "dev-" + region
        else:
            raise ValueError("Environment must be 'dev' or 'perf'.")
        run = self._start_deployment(
            instance=instance or self.instance,
            job=job,
            disk_folder=disk_folder,
            application_zip_bytes=data,
            version=version,
        )
        self._follow_deployment(instance, job, run)
        app: Vespa = self.get_application(
            instance=instance or self.instance,
            environment=environment,
            endpoint_type="mtls",
        )
        return app

    def delete(
        self, instance: Optional[str] = "default", environment: Optional[str] = "dev"
    ) -> None:
        """
        Delete the specified instance from the development environment in the Vespa Cloud.

        To delete a production instance, you must submit a new deployment with `deployment-removal` added to the 'validation-overrides.xml'.
        See <https://cloud.vespa.ai/en/deleting-applications> for more details.

        Example usage:
            ```python
            vespa_cloud = VespaCloud(...)
            vespa_cloud.delete_instance(instance="my-instance")
            ```

        Args:
            instance (str): The name of the instance to delete.
            environment (str): The environment from which to delete the instance. Must be "dev" or "perf".

        Returns:
            None
        """
        if environment == "dev":
            region = self.get_dev_region()
        elif environment == "perf":
            region = self.get_perf_region()
        else:
            raise ValueError("Environment must be 'dev' or 'perf'.")
        print(
            self._request(
                "DELETE",
                "/application/v4/tenant/{}/application/{}/instance/{}/environment/{}/region/{}".format(
                    self.tenant, self.application, instance, environment, region
                ),
            )["message"],
            file=self.output,
        )

    @staticmethod
    def _read_private_key(
        key_location: Optional[str] = None, key_content: Optional[str] = None
    ) -> Optional[ec.EllipticCurvePrivateKey]:
        if key_content:
            key_content = bytes(key_content, "ascii")
        elif key_location:
            with open(key_location, "rb") as key_data:
                key_content = key_data.read()
        else:
            return None

        key = serialization.load_pem_private_key(key_content, None, default_backend())
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise TypeError("Key must be an elliptic curve private key")
        return key

    def _check_vespacli_available(self) -> bool:
        try:
            vespa_version = subprocess.run(
                shlex.split("vespa version"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.decode("utf-8")
            is_available = "Vespa CLI" in vespa_version
        except FileNotFoundError:
            print(
                "Vespa CLI not found. Run `pip install vespacli`.",
            )
            is_available = False
        return is_available

    def _set_target_cloud(self):
        print("Running: vespa config set target cloud")
        output = subprocess.run(
            shlex.split("vespa config set target cloud"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if output.returncode != 0:
            raise RuntimeError(
                f"Failed to set target cloud with 'vespa config set target cloud'. Return code: {output.returncode}"
            )
        else:
            print(output.stdout.decode("utf-8"))

    def _vespa_auth_login(self):
        system = platform.system()
        if system == "Windows":
            raise NotImplementedError(
                "Interactive login is not supported on Windows due to unavailable `pty` module. Please use 'vespa auth login' in the terminal instead."
            )
        import pty

        # Open a new pseudo-terminal
        master, slave = pty.openpty()

        # Start the subprocess with its input/output connected to the PTY
        p = subprocess.Popen(
            shlex.split("vespa auth login"),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            universal_newlines=True,
        )

        # Close the slave end in the parent process
        os.close(slave)
        finished = False
        try:
            while not finished:
                # Use select to wait for data to be available on the PTY
                rlist, _, _ = select.select([master], [], [], 1)

                for fd in rlist:
                    if fd == master:
                        # Read output from the master end of the PTY
                        output = os.read(master, 1024).decode("utf-8")
                        if output:
                            print(output, end="")
                            sys.stdout.flush()
                        if "Success:" in output:
                            finished = True  # Exit the loop after success message
                            break

                        # Check for input if running in a Jupyter Notebook or terminal
                        if "[Y/n]" in output:
                            user_input = input() + "\n"
                            os.write(master, user_input.encode())
                            sys.stdout.flush()
                if finished:
                    break

        finally:
            # Ensure the master end of the PTY is closed
            os.close(master)
            # Ensure the subprocess is properly terminated
            p.terminate()
            p.wait()
        #
        auth_json_path = VESPA_HOME / "auth.json"
        while not auth_json_path.exists():
            sleep(1)
        print(f" auth.json created at {auth_json_path}")
        return

    def _vespa_auth_cert(self):
        print("Running: vespa auth cert -N")
        output = subprocess.run(
            shlex.split("vespa auth cert -N"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # This may return a non-zero exit code if the certificate and key already exist
        print(output.stdout.decode("utf-8"))

    def _set_application(self):
        vespa_cli_command = f"vespa config set application {self.tenant}.{self.application}.{self.instance}"
        print("Running: " + vespa_cli_command)
        output = subprocess.run(
            shlex.split(vespa_cli_command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if output.returncode != 0:
            raise RuntimeError(
                f"Failed to set application with '{vespa_cli_command}'. Return code: {output.returncode}"
            )

    def _load_certificate_pair(
        self,
        generate_cert: bool = True,  # Need a flag to avoid infinite recursion
    ) -> Tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
        def _check_dir(path: Path):
            cert_path = path / "data-plane-public-cert.pem"
            key_path = path / "data-plane-private-key.pem"

            if cert_path.exists() and key_path.exists():
                return str(cert_path), str(key_path)
            else:
                return None, None

        # Try to look in application root first, assuming working directory is the same as application root
        local_vespa_dir = Path.cwd() / ".vespa"
        cert, key = _check_dir(local_vespa_dir)
        if cert and key:
            self.data_cert_path = cert
            self.data_key_path = key
        else:
            # If cert/key not found in application root: look in ~/.vespa/tenant.app.default/
            home_vespa_dir = (
                VESPA_HOME / f"{self.tenant}.{self.application}.{self.instance}"
            )
            cert, key = _check_dir(home_vespa_dir)
            if cert and key:
                self.data_cert_path = cert
                self.data_key_path = key

        if self.data_key_path and self.data_cert_path:
            # Cert and key generated with 'vespa auth cert' were successfully found somewhere

            # Read contents of private key from file
            with open(self.data_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(), password=None, backend=default_backend()
                )
            # Read contents of public certificate from file
            with open(self.data_cert_path, "rb") as cert_file:
                cert = x509.load_pem_x509_certificate(
                    cert_file.read(), default_backend()
                )
            return private_key, cert
        else:
            if generate_cert:
                if self._check_vespacli_available():
                    print(
                        f"Certificate and key not found in {local_vespa_dir} or {home_vespa_dir}: Creating new cert/key pair with vespa CLI."
                    )
                    self._generate_cert_vespacli()
                    return self._load_certificate_pair(generate_cert=False)
        raise FileNotFoundError(
            "Certificate and key not found in ~/.vespa or .vespa. \n Unable to generate as vespa CLI not detected. Please install vespacli (`pip install vespacli`) or generate a cert/key pair manually with 'vespa auth cert -N'."
        )

    def _generate_cert_vespacli(self) -> None:
        # Run vespa auth cert
        print("Generating certificate and key...")
        self._vespa_auth_cert()
        return

    def get_dev_region(self) -> str:
        return "aws-us-east-1c"  # Default dev region

    def get_perf_region(self) -> str:
        # Only one available for now (https://cloud.vespa.ai/en/reference/zones)
        return "aws-us-east-1c"  # Default perf region

    def get_prod_region(self):
        regions = self.get_prod_regions()
        return regions[0]

    def get_prod_regions(self, instance: Optional[str] = "default") -> List[str]:
        regions = []
        info = self._request(
            method="GET",
            path=f"/application/v4/tenant/{self.tenant}/application/{self.application}",
        )
        for inst in info["instances"]:
            if inst["instance"] == instance:
                for deployment in inst["deployments"]:
                    if deployment["environment"] == "prod":
                        regions.append(deployment["region"])
        if not regions:
            raise ValueError(
                f"No production regions found for instance {instance}, available instances: {info['instances']}",
            )
        return regions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=3),
        reraise=True,
    )
    def get_connection_response_with_retry(
        self,
        method,
        path,
        body: Optional[Union[BytesIO, Dict]] = None,
        headers: Dict = {},
    ) -> httpx.Response:
        if isinstance(body, dict):
            data = body
            content = None
        elif isinstance(body, BytesIO):
            data = None
            content = body.getvalue()
        else:
            data = None
            content = None
        with (
            httpx.Client(
                base_url=self.base_url,
                headers=self.base_headers,
                timeout=None,  # Need to set timeout to None to avoid httpx timeout on e.g. deployment requests
                http1=True,
                limits=self.httpx_limits,
            ) as client
        ):
            response = client.request(
                method, path, data=data, content=content, headers=headers
            )
            if response.status_code != 200:
                raise HTTPError(
                    f"HTTP {response.status_code} reason: {response.reason_phrase} error_text: {response.text} for {path}"
                )
        return response

    def _try_get_access_token(self) -> str:
        # Check if auth.json exists
        if not self.auth_file_path.exists():
            print("No auth.json found. Please authenticate.")
            self._vespa_auth_login()
            # Recheck for auth.json after authentication
            if not self.auth_file_path.exists():
                raise FileNotFoundError("Authentication failed, auth.json not found.")
        else:
            print("Checking for access token in auth.json...")

        # Load the auth.json file
        auth = json.loads(self.auth_file_path.read_text())

        # Ensure the datetime string is parsed correctly
        try:
            expires_at = parser.parse(
                auth["providers"]["auth0"]["systems"]["public"]["expires_at"]
            )
        except ValueError as e:
            print(f"Error parsing the date: {e}")
            raise

        # Compare offset-aware datetime objects
        if expires_at < datetime.now(timezone.utc):
            print("Access token expired. Please re-authenticate.")
            # Remove the expired file
            os.remove(self.auth_file_path)
            self._vespa_auth_login()
            # Reload the auth.json file after re-authentication
            auth = json.loads(self.auth_file_path.read_text())
            try:
                expires_at = parser.parse(
                    auth["providers"]["auth0"]["systems"]["public"]["expires_at"]
                )
            except ValueError as e:
                print(f"Error parsing the date: {e}")
                raise
            if expires_at < datetime.now(timezone.utc):
                raise Exception("Authentication failed, token is still expired.")

        return auth["providers"]["auth0"]["systems"]["public"]["access_token"]

    def _handle_response(
        self,
        response: httpx.Response,
        return_raw_response: bool = False,
        path: str = "",
    ) -> Union[dict, httpx.Response]:
        """Common response handling logic"""
        if return_raw_response:
            return response

        try:
            parsed = json.load(response)
        except json.JSONDecodeError:
            parsed = response.read()

        if response.status_code != 200:
            print(parsed)
            raise HTTPError(
                f"HTTP {response.status_code} error: {response.reason_phrase} for {path}"
            )
        return parsed

    def _get_auth_headers(self, additional_headers: dict = {}) -> dict:
        """Create authorization headers"""
        if not self.control_plane_access_token:
            raise ValueError("Access token not set.")

        return {
            "Authorization": f"Bearer {self.control_plane_access_token}",
            **additional_headers,
        }

    def _request_with_access_token(
        self,
        method: str,
        path: str,
        body: Union[BytesIO, MultipartEncoder] = BytesIO(),
        headers: dict = {},
        return_raw_response: bool = False,
    ) -> Union[dict, httpx.Response]:
        """Make authenticated request with access token"""
        if hasattr(body, "seek"):
            body.seek(0)

        auth_headers = self._get_auth_headers(headers)
        response = self.get_connection_response_with_retry(
            method, path, body, auth_headers
        )

        return self._handle_response(response, return_raw_response, path)

    def _request(
        self,
        method: str,
        path: str,
        body: Union[BytesIO, MultipartEncoder] = BytesIO(),
        headers: dict = {},
        return_raw_response: bool = False,
    ) -> Union[dict, httpx.Response]:
        if self.control_plane_auth_method == "access_token":
            return self._request_with_access_token(method, path, body, headers)
        elif self.control_plane_auth_method == "api_key":
            return self._request_with_api_key(
                method, path, body, headers, return_raw_response
            )
        else:
            raise ValueError(
                "Control plane auth method not inferred. Should be either api_key or access_token."
            )

    def _request_with_api_key(
        self,
        method: str,
        path: str,
        body: Union[BytesIO, MultipartEncoder] = BytesIO(),
        headers: dict = {},
        return_raw_response: bool = False,
    ) -> Union[dict, httpx.Response]:
        digest = hashes.Hash(hashes.SHA256(), default_backend())

        # Handle different body types
        if isinstance(body, MultipartEncoder):
            # Use the encoded data for hash computation
            digest = hashes.Hash(hashes.SHA256(), default_backend())
            digest.update(body.to_string())  # This moves the buffer position to the end
            body._buffer.seek(0)  # Needs to be reset. Otherwise, no data will be sent
            # Update the headers to include the Content-Type
            headers.update({"Content-Type": body.content_type})
            # Read the content of multipart_data into a bytes object
            multipart_data_bytes: bytes = body.to_string()
            headers.update({"Content-Length": str(len(multipart_data_bytes))})
            # Convert multipart_data_bytes to type BytesIO
            body_data: BytesIO = BytesIO(multipart_data_bytes)
        else:
            if hasattr(body, "seek"):
                body.seek(0)
            digest.update(body.read())
            body_data = body
        # Create signature
        content_hash = standard_b64encode(digest.finalize()).decode("UTF-8")
        timestamp = datetime.utcnow().isoformat() + "Z"
        url = self.base_url + path

        canonical_message = method + "\n" + url + "\n" + timestamp + "\n" + content_hash
        signature = self.api_key.sign(
            canonical_message.encode("UTF-8"), ec.ECDSA(hashes.SHA256())
        )
        signature_b64 = standard_b64encode(signature).decode("UTF-8")

        headers = {
            "X-Timestamp": timestamp,
            "X-Content-Hash": content_hash,
            "X-Key-Id": f"{self.tenant}:{self.application}:default",
            "X-Key": self.api_public_key_bytes,
            "X-Authorization": signature_b64,
            **headers,
        }

        response = self.get_connection_response_with_retry(
            method, path, body_data, headers
        )
        return self._handle_response(response, return_raw_response, path)

    def get_all_endpoints(
        self,
        instance: Optional[str] = "default",
        region: Optional[str] = None,
        environment: Optional[str] = "dev",
    ) -> List[Dict[str, str]]:
        """
        Get all endpoints for the application instance.

        Args:
            instance (str): Application instance name.
            region (str): Region name, e.g. 'aws-us-east-1c'.
            environment (str): Environment (dev/perf/prod).

        Returns:
            list: List of endpoints.
        """

        if region is None:
            if environment == "dev":
                region = self.get_dev_region()
            elif environment == "perf":
                region = self.get_perf_region()
            elif environment == "prod":
                region = self.get_prod_region()
            else:
                raise ValueError(
                    "Invalid environment. Must be 'dev', 'perf', or 'prod'"
                )

        endpoints = self._request(
            "GET",
            "/application/v4/tenant/{}/application/{}/instance/{}/environment/{}/region/{}".format(
                self.tenant, self.application, instance, environment, region
            ),
        )["endpoints"]
        return endpoints

    def get_app_package_contents(
        self,
        instance: Optional[str] = "default",
        region: Optional[str] = None,
        environment: Optional[str] = "dev",
    ) -> List[str]:
        """
        Get all endpoints for the application package content in the specified region and environment.

        Args:
            instance (str): Application instance name.
            region (str): Region name, e.g. 'aws-us-east-1c'. If None, uses the default region for the environment.
            environment (str): Environment (dev/perf/prod). Default is 'dev'.

        Returns:
            list: List of endpoints for the application instance.
        """
        if region is None:
            if environment == "dev":
                region = self.get_dev_region()
            elif environment == "perf":
                region = self.get_perf_region()
            elif environment == "prod":
                region = self.get_prod_region()
            else:
                raise ValueError(
                    "Invalid environment. Must be 'dev', 'perf', or 'prod'"
                )

        app_endpoints = self._request(
            "GET",
            "/application/v4/tenant/{}/application/{}/instance/{}/environment/{}/region/{}/content/".format(
                self.tenant, self.application, instance, environment, region
            ),
        )
        return app_endpoints

    def get_schemas(
        self,
        instance: Optional[str] = "default",
        region: Optional[str] = None,
        environment: Optional[str] = "dev",
    ) -> Dict[str, str]:
        """
        Get all schemas for the application instance in the specified environment and region.

        Args:
            instance (str): Application instance name.
            region (str): Region name, e.g. 'aws-us-east-1c'. If None, uses the default region for the environment.
            environment (str): Environment (dev/perf/prod). Default is 'dev'.

        Returns:
            dict: Dictionary with schema name as key and content as value.
        """
        if region is None:
            if environment == "dev":
                region = self.get_dev_region()
            elif environment == "perf":
                region = self.get_perf_region()
            elif environment == "prod":
                region = self.get_prod_region()
            else:
                raise ValueError(
                    "Invalid environment. Must be 'dev', 'perf', or 'prod'"
                )
        schemas = {
            schema_endpoint.split("/")[-1][:-3]: self._request(
                "GET", urlparse(schema_endpoint).path
            ).decode("utf-8")
            for schema_endpoint in self._request(
                "GET",
                "/application/v4/tenant/{}/application/{}/instance/{}/environment/{}/region/{}/content/schemas/".format(
                    self.tenant, self.application, instance, environment, region
                ),
            )
        }

        return schemas

    def download_app_package_content(
        self,
        destination_path: str,
        instance: Optional[str] = "default",
        region: Optional[str] = None,
        environment: Optional[str] = "dev",
    ) -> None:
        """
        Download the application package content to a specified destination path.

        Args:
            destination_path (str): The path where the application package content will be downloaded.
            instance (str): Application instance name.
            region (str): Region name, e.g. 'aws-us-east-1c'. If None, uses the default region for the environment.
            environment (str): Environment (dev/perf/prod). Default is 'dev'.

        Returns:
            None
        """
        if region is None:
            if environment == "dev":
                region = self.get_dev_region()
            elif environment == "perf":
                region = self.get_perf_region()
            elif environment == "prod":
                region = self.get_prod_region()
            else:
                raise ValueError(
                    "Invalid environment. Must be 'dev', 'perf', or 'prod'"
                )

        for endpoint in self.get_app_package_contents(instance, region, environment):
            path = urlparse(endpoint).path
            relative_path = path.split("content/")[-1]
            file_path = os.path.expanduser(
                os.path.join(destination_path, relative_path)
            )  # Path to donwload to

            if path.endswith("/"):
                continue  # Skip directories

            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            try:
                data = self._request(
                    "GET", urlparse(endpoint).path, return_raw_response=False
                )

                with open(file_path, "wb") as f:
                    f.write(data)
            except Exception as e:
                print(f"Error downloading {endpoint}: {e}")

    def get_endpoint_auth_method(
        self,
        url: str,
        instance: Optional[str] = "default",
        region: Optional[str] = None,
        environment: Optional[str] = "dev",
    ) -> str:
        """
        Get the authentication method for the given endpoint URL.

        Args:
            url (str): The endpoint URL.
            instance (str): Application instance name.
            region (str): Region name, e.g. 'aws-us-east-1c'.
            environment (str): Environment (dev/perf/prod).

        Returns:
            str: The authentication method ('mtls' or 'token').
        """

        endpoints = self.get_all_endpoints(instance, region, environment)
        for endpoint in endpoints:
            if endpoint["url"] == url:
                return endpoint["authMethod"]
        else:
            return None

    def get_endpoint(
        self,
        auth_method: str,
        instance: Optional[str] = "default",
        region: Optional[str] = None,
        environment: Optional[str] = "dev",
        cluster: Optional[str] = None,
    ) -> str:
        """
        Get the endpoint URL for the application.

        Tip: See the 'endpoint'-tab in Vespa Cloud Console for available endpoints.

        Args:
            auth_method (str): Authentication method. Options are 'mtls' or 'token'.
            instance (str): Application instance name.
            region (str): Region name, e.g. 'aws-us-east-1c'.
            environment (str): Environment (dev/perf/prod).
            cluster (str): Specific cluster to get the endpoint for. If None, uses the instance's default cluster.

        Returns:
            str: The endpoint URL.
        """
        cluster = cluster or self.cluster
        auth_endpoints = []
        available_clusters = set()
        endpoints = self.get_all_endpoints(instance, region, environment)

        for endpoint in endpoints:
            endpoint_cluster = endpoint.get("cluster", None)
            available_clusters.add(endpoint_cluster)

            if cluster is not None and cluster != endpoint_cluster:
                continue

            authMethod = endpoint.get("authMethod", None)
            if authMethod == auth_method:
                print(
                    f"Found {auth_method} endpoint for {endpoint_cluster}",
                    file=self.output,
                )
                print(f"URL: {endpoint['url']}", file=self.output)
                auth_endpoints.append((endpoint["url"], endpoint_cluster))

        if len(auth_endpoints) == 0:
            error_msg = f"No {auth_method} endpoints found for instance {instance}, region {region}, and environment {environment}"
            if cluster:
                error_msg += f"\nRequested cluster '{cluster}' not found. Available clusters: {sorted(available_clusters)}"
            raise RuntimeError(error_msg)

        if len(auth_endpoints) > 1:
            if cluster:
                matching = [
                    url for url, ep_cluster in auth_endpoints if ep_cluster == cluster
                ]
                if matching:
                    return matching[0]
            print(
                f"Multiple {auth_method} endpoints found. Available clusters: {sorted(c for _, c in auth_endpoints)}",
                file=self.output,
            )
            print("Returning the first endpoint.", file=self.output)

        return auth_endpoints[0][0]

    def get_mtls_endpoint(
        self,
        instance: Optional[str] = "default",
        region: Optional[str] = None,
        environment: Optional[str] = "dev",
        cluster: Optional[str] = None,
    ) -> str:
        """
        Get the endpoint URL of a mTLS endpoint for the application.
        Will return the first mTLS endpoint found if multiple exist.
        Use `VespaCloud.get_all_endpoints` to get all endpoints.

        Tip: See the 'endpoint'-tab in Vespa Cloud Console for available endpoints.

        Args:
            instance (str): Application instance name.
            region (str): Region name.
            environment (str): Environment (dev/perf/prod).
            cluster (str): Specific cluster to get the endpoint for. If None, uses the instance's default cluster.

        Returns:
            str: The endpoint URL.
        """

        return self.get_endpoint("mtls", instance, region, environment, cluster)

    def get_token_endpoint(
        self,
        instance: Optional[str] = "default",
        region: Optional[str] = None,
        environment: Optional[str] = "dev",
        cluster: Optional[str] = None,
    ) -> str:
        """
        Get the endpoint URL of a token endpoint for the application.
        Will return the first token endpoint found if multiple exist.
        Use `VespaCloud.get_all_endpoints` to get all endpoints.

        Tip: See the 'endpoint'-tab in Vespa Cloud Console for available endpoints.

        Args:
            instance (str): Application instance name.
            region (str): Region name.
            environment (str): Environment (dev/perf/prod).
            cluster (str): Specific cluster to get the endpoint for. If None, uses the instance's default cluster.

        Returns:
            str: The endpoint URL.
        """
        return self.get_endpoint("token", instance, region, environment, cluster)

    def _application_root_has_tests(self, application_root: str) -> bool:
        """Check if the application contains tests folder (recursively)"""
        for root, dirs, files in os.walk(application_root):
            if "tests" in dirs:
                return True
        return False

    def _write_certificate(self, cert_path: str, overwrite: bool = False) -> None:
        if not os.path.exists(cert_path) or overwrite:
            os.makedirs(os.path.dirname(cert_path), exist_ok=True)
            with open(cert_path, "wb") as clients_pem:
                clients_pem.write(
                    self.data_certificate.public_bytes(serialization.Encoding.PEM)
                )
        elif os.path.exists(cert_path):
            logging.info(
                f"Will use existing file at {cert_path}."
                f"Remove it to add new clients.pem-file on deployment."
            )

    def _start_prod_deployment(
        self, application_root: str, source_url: str = "", instance: str = "default"
    ) -> int:
        # The submit API is used for prod deployments
        deploy_path = "/application/v4/tenant/{}/application/{}/submit/".format(
            self.tenant, self.application
        )

        # Create app package zip
        Path(application_root).mkdir(parents=True, exist_ok=True)
        if self.application_package is not None:
            application_package_zip_bytes = self._to_application_zip(
                disk_folder=application_root
            )
        else:
            # Need to write the certificate to disk_folder in security/clients.pem
            client_pem_path = os.path.join(application_root, "security/clients.pem")
            self._write_certificate(cert_path=client_pem_path)
            application_package_zip_bytes = BytesIO(
                self.read_app_package_from_disk(application_root)
            )

        submit_options = {
            "projectId": 1,
            "risk": 0,
        }
        if source_url:
            submit_options["sourceUrl"] = source_url

        # Define the fields data to add to the multipart data
        fields = {
            "submitOptions": ("", json.dumps(submit_options), "application/json"),
            "applicationZip": (
                "application.zip",
                application_package_zip_bytes,
                "application/zip",
            ),
        }

        parent_path = Path(application_root).parent
        application_test_path = parent_path.joinpath("application-test")

        # Check if the application contains tests folder. If so, submit as application-test.zip
        if self._application_root_has_tests(application_root):
            test_fields = {
                "applicationTestZip": (
                    "application-test.zip",
                    application_package_zip_bytes,  # Just submitting duplicate of application package (same behavior as Vespa CLI)
                    "application/zip",
                )
            }
            fields.update(test_fields)
        elif (
            application_test_path.exists()
        ):  # If packaged with maven, use the existing application-test
            print(
                f"`application-test´ found in {parent_path}. Including in package.",
                file=self.output,
            )
            test_zip_bytes = BytesIO(
                self.read_app_package_from_disk(application_test_path)
            )
            test_fields = {
                "applicationTestZip": (
                    "application-test.zip",
                    test_zip_bytes,
                    "application/zip",
                )
            }
            fields.update(test_fields)
        else:
            print(
                f"No `tests` folder found in {application_root}. No `application-test` directory found in {parent_path}.",
                file=self.output,
            )
            print("No tests will be submitted.", file=self.output)
        multipart_data = MultipartEncoder(
            fields=fields,
        )

        # Compute content hash, etc
        if self.control_plane_auth_method == "api_key":
            url = self.base_url + deploy_path
            digest = hashes.Hash(hashes.SHA256(), default_backend())
            digest.update(
                multipart_data.to_string()
            )  # This moves the buffer position to the end
            multipart_data._buffer.seek(
                0
            )  # Needs to be reset. Otherwise, no data will be sent
            content_hash = standard_b64encode(digest.finalize()).decode("UTF-8")
            timestamp = (
                datetime.utcnow().isoformat() + "Z"
            )  # Java's Instant.parse requires the neutral time zone appended
            canonical_message = (
                "POST" + "\n" + url + "\n" + timestamp + "\n" + content_hash
            )
            signature = self.api_key.sign(
                canonical_message.encode("UTF-8"), ec.ECDSA(hashes.SHA256())
            )
            headers = {
                "X-Timestamp": timestamp,
                "X-Content-Hash": content_hash,
                "X-Key-Id": self.tenant + ":" + self.application + ":" + instance,
                "X-Key": self.api_public_key_bytes,
                "X-Authorization": standard_b64encode(signature),
            }
        elif self.control_plane_auth_method == "access_token":
            headers = {
                "Authorization": "Bearer " + self.control_plane_access_token,
            }
        # Read the content of multipart_data into a bytes object
        multipart_data_bytes: bytes = multipart_data.to_string()
        headers["Content-Length"] = str(len(multipart_data_bytes))
        # Update the headers to include the Content-Type
        headers["Content-Type"] = multipart_data.content_type
        # Convert multipart_data_bytes to type BytesIO
        multipart_data_bytes = BytesIO(multipart_data_bytes)
        response = self._request(
            "POST", deploy_path, body=multipart_data_bytes, headers=headers
        )
        message = response.get("message", "No message provided")
        print(message, file=self.output)
        build_no = int(response.get("build"))
        skipped = bool(response.get("skipped"))
        if skipped:
            deployable_build_no = self._get_last_deployable(build_no)
            print(
                f"Build {build_no} will not be deployed, being equal to the previous one. Returning last deployable build {deployable_build_no} instead.",
                file=self.output,
            )
            build_no = deployable_build_no
        # Set submitted_timestamp in format 1718776065383
        # Use current UTC time in milliseconds
        self.submitted_timestamp = int(datetime.utcnow().timestamp() * 1000)
        return build_no

    def _start_deployment(
        self,
        instance: str,
        job: str,
        disk_folder: str,
        application_zip_bytes: Optional[BytesIO] = None,
        version: Optional[str] = None,
    ) -> int:
        deploy_path = (
            "/application/v4/tenant/{}/application/{}/instance/{}/deploy/{}".format(
                self.tenant, self.application, instance, job
            )
        )

        if version is not None:
            # Create multipart form data
            form_data = {
                "applicationZip": (
                    "application.zip",
                    application_zip_bytes,
                    "application/zip",
                ),
                "deployOptions": (
                    "",
                    json.dumps({"vespaVersion": version}),
                    "application/json",
                ),
            }
            multipart = MultipartEncoder(fields=form_data)
            headers = {"Content-Type": multipart.content_type}
            payload = multipart
        else:
            # Use existing direct zip upload
            headers = {"Content-Type": "application/zip"}
            payload = application_zip_bytes

        response = self._request(
            method="POST", path=deploy_path, body=payload, headers=headers
        )
        message = response.get("message", "No message provided")
        print(message, file=self.output)
        return response["run"]

    def _to_application_zip(self, disk_folder: str) -> BytesIO:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "a") as zip_archive:
            for schema in self.application_package.schemas:
                zip_archive.writestr(
                    "schemas/{}.sd".format(schema.name),
                    schema.schema_to_text,
                )
                for model in schema.models:
                    zip_archive.write(
                        model.model_file_path,
                        os.path.join("files", model.model_file_name),
                    )

            if self.application_package.models:
                for model_id, model in self.application_package.models.items():
                    temp_model_file = os.path.join(
                        disk_folder,
                        "{}.onnx".format(model_id),
                    )
                    model.export_to_onnx(output_path=temp_model_file)
                    zip_archive.write(
                        temp_model_file,
                        "models/{}.onnx".format(model_id),
                    )
                    os.remove(temp_model_file)

            if self.application_package.query_profile:
                zip_archive.writestr(
                    "search/query-profiles/default.xml",
                    self.application_package.query_profile_to_text,
                )
                zip_archive.writestr(
                    "search/query-profiles/types/root.xml",
                    self.application_package.query_profile_type_to_text,
                )
            zip_archive.writestr(
                "services.xml", self.application_package.services_to_text
            )
            zip_archive.writestr(
                "security/clients.pem",
                self.data_certificate.public_bytes(serialization.Encoding.PEM),
            )
            if self.application_package.deployment_config:
                zip_archive.writestr(
                    "deployment.xml", self.application_package.deployment_to_text
                )
            if self.application_package.validations:
                zip_archive.writestr(
                    "validation-overrides.xml",
                    self.application_package.validations_to_text,
                )

        return buffer

    def _follow_deployment(
        self, instance: str, job: str, run: int, max_wait: int = 1800
    ) -> None:
        last = -1
        start = time.time()
        while time.time() - start < max_wait:
            try:
                status, last = self._get_deployment_status(instance, job, run, last)
            except RuntimeError:
                raise

            if status == "active":
                sleep(1)
            elif status == "success":
                return
            else:
                raise RuntimeError("Unexpected status: {}".format(status))
        raise TimeoutError(f"Deployment did not finish within {max_wait} seconds.")

    def _get_deployment_status(
        self, instance: str, job: str, run: int, last: int
    ) -> Tuple[str, int]:
        update = self._request(
            "GET",
            "/application/v4/tenant/{}/application/{}/instance/{}/job/{}/run/{}?after={}".format(
                self.tenant, self.application, instance, job, run, last
            ),
        )

        for step, entries in update["log"].items():
            for entry in entries:
                self._print_log_entry(step, entry)
        last = update.get("lastId", last)

        fail_status_message = {
            "error": "Unexpected error during deployment; see log for details",
            "aborted": "Deployment was aborted, probably by a newer deployment",
            "outOfCapacity": "No capacity left in zone; please contact the Vespa team",
            "deploymentFailed": "Deployment failed; see log for details",
            "installationFailed": "Installation failed; see Vespa log for details",
            "running": "Deployment not completed",
            "endpointCertificateTimeout": "Endpoint certificate not ready in time; please contact Vespa team",
            "testFailure": "Unexpected status; tests are not run for manual deployments",
        }

        if update["active"]:
            return "active", last
        else:
            status = update["status"]
            if status == "success":
                return "success", last
            if status == "noTests":
                # We'll proceed as usual for now, as this is allowed.
                # In the future, we should support tests via Pyvespa properly, though.
                # TODO Support tests via Pyvespa
                return "success", last
            elif status in fail_status_message.keys():
                raise RuntimeError(fail_status_message[status])
            else:
                raise RuntimeError("Unexpected status: {}".format(status))

    def _print_log_entry(self, step: str, entry: dict):
        timestamp = strftime("%H:%M:%S", gmtime(entry["at"] / 1e3))
        message = entry["message"].replace("\n", "\n" + " " * 23)
        if step != "copyVespaLogs" or entry["type"] == "error":
            print(
                "{:<7} [{}]  {}".format(entry["type"].upper(), timestamp, message),
                file=self.output,
            )
