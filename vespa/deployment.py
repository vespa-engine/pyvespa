import http.client
import json
import os
import re
import sys
import zipfile
from base64 import standard_b64encode
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from shutil import copyfile
from time import sleep, strftime, gmtime
from typing import Union, IO, Optional, Mapping

import docker
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec

from vespa.application import Vespa
from vespa.json_serialization import ToJson, FromJson
from vespa.package import ApplicationPackage, ModelServer


class VespaDocker(ToJson, FromJson["VespaDocker"]):
    def __init__(
        self,
        disk_folder: Optional[str] = None,
        port: int = 8080,
        container_memory: Union[str, int] = 4 * (1024 ** 3),
        output_file: IO = sys.stdout,
        container: Optional[docker.models.containers.Container] = None,
    ) -> None:
        """
        Manage Docker deployments.

        :param disk_folder: Disk folder to save the required Vespa config files. Default to application name
            folder within user's current working directory.
        :param port: Container port.
        :param output_file: Output file to write output messages.
        :param container_memory: Docker container memory available to the application.
        :param container: Used when instantiating VespaDocker from a running container.
        """
        self.container = container
        container_id = None
        container_name = None
        if container:
            container_id = container.id
            container_name = container.name
        self.container_name = container_name
        self.container_id = container_id
        self.url = "http://localhost"
        self.local_port = port
        self.disk_folder = disk_folder
        self.container_memory = container_memory
        self.output = output_file

    @staticmethod
    def from_container_name_or_id(
        name_or_id: str, output_file: IO = sys.stdout
    ) -> "VespaDocker":
        """
        Instantiate VespaDocker from a running container.

        :param name_or_id: Name or id of the container.
        :param output_file: Output file to write output messages.
        :return: VespaDocker instance associated with the running container.
        """
        client = docker.from_env()
        try:
            container = client.containers.get(name_or_id)
        except docker.errors.NotFound:
            raise ValueError("The container does not exist.")
        disk_folder = container.attrs["Mounts"][0]["Source"]
        port = int(
            container.attrs["HostConfig"]["PortBindings"]["8080/tcp"][0]["HostPort"]
        )
        container_memory = container.attrs["HostConfig"]["Memory"]

        return VespaDocker(
            disk_folder=disk_folder,
            port=port,
            container_memory=container_memory,
            output_file=output_file,
            container=container,
        )

    def _run_vespa_engine_container(
        self,
        application_name: str,
        disk_folder: str,
        container_memory: str,
    ):
        client = docker.from_env()
        if self.container is None:
            try:
                self.container = client.containers.get(application_name)
                self.container.restart()
            except docker.errors.NotFound:
                self.container = client.containers.run(
                    "vespaengine/vespa",
                    detach=True,
                    mem_limit=container_memory,
                    name=application_name,
                    hostname=application_name,
                    privileged=True,
                    volumes={disk_folder: {"bind": "/app", "mode": "rw"}},
                    ports={8080: self.local_port},
                )
            self.container_name = self.container.name
            self.container_id = self.container.id
        else:
            self.container.restart()

    def _check_configuration_server(self) -> bool:
        """
        Check if configuration server is running and ready for deployment
        :return: True if configuration server is running.
        """
        return (
            self.container is not None
            and self.container.exec_run(
                "bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'"
            )
            .output.decode("utf-8")
            .split("\r\n")[0]
            == "HTTP/1.1 200 OK"
        )

    def export_application_package(
        self, application_package: Union[ApplicationPackage, ModelServer]
    ) -> None:
        """
        Export application package to disk.
        :param application_package: Application package to export.
        :return: None. Application package file will be stored on `disk_folder`.
        """
        if not self.disk_folder:
            self.disk_folder = os.path.join(os.getcwd(), application_package.name)

        Path(os.path.join(self.disk_folder, "application/schemas")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.disk_folder, "application/files")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.disk_folder, "application/models")).mkdir(
            parents=True, exist_ok=True
        )

        for schema in application_package.schemas:
            with open(
                os.path.join(
                    self.disk_folder,
                    "application/schemas/{}.sd".format(schema.name),
                ),
                "w",
            ) as f:
                f.write(schema.schema_to_text)
            for model in schema.models:
                copyfile(
                    model.model_file_path,
                    os.path.join(
                        self.disk_folder, "application/files", model.model_file_name
                    ),
                )

        if application_package.query_profile:
            Path(
                os.path.join(
                    self.disk_folder, "application/search/query-profiles/types"
                )
            ).mkdir(parents=True, exist_ok=True)
            with open(
                os.path.join(
                    self.disk_folder,
                    "application/search/query-profiles/default.xml",
                ),
                "w",
            ) as f:
                f.write(application_package.query_profile_to_text)
            with open(
                os.path.join(
                    self.disk_folder,
                    "application/search/query-profiles/types/root.xml",
                ),
                "w",
            ) as f:
                f.write(application_package.query_profile_type_to_text)
        with open(os.path.join(self.disk_folder, "application/hosts.xml"), "w") as f:
            f.write(application_package.hosts_to_text)
        with open(os.path.join(self.disk_folder, "application/services.xml"), "w") as f:
            f.write(application_package.services_to_text)

        if application_package.models:
            for model_id, model in application_package.models.items():
                model.export_to_onnx(
                    output_path=os.path.join(
                        self.disk_folder,
                        "application/models/{}.onnx".format(model_id),
                    )
                )

    def _execute_deployment(
        self,
        application_name: str,
        disk_folder: str,
        container_memory: str = "4G",
        application_folder: Optional[str] = None,
        application_package: Optional[ApplicationPackage] = None,
    ):

        self._run_vespa_engine_container(
            application_name=application_name,
            disk_folder=disk_folder,
            container_memory=container_memory,
        )

        while not self._check_configuration_server():
            print("Waiting for configuration server.", file=self.output)
            sleep(5)

        _application_folder = "/app"
        if application_folder:
            _application_folder = (
                _application_folder + "/" + application_folder
            )  # using os.path.join break on windows
        deployment = self.container.exec_run(
            "bash -c '/opt/vespa/bin/vespa-deploy prepare {} && /opt/vespa/bin/vespa-deploy activate'".format(
                _application_folder
            )
        )

        deployment_message = deployment.output.decode("utf-8").split("\n")

        if not any(re.match("Generation: [0-9]+", line) for line in deployment_message):
            raise RuntimeError(deployment_message)

        app = Vespa(
            url=self.url,
            port=self.local_port,
            deployment_message=deployment_message,
            application_package=application_package,
        )

        while not app.get_application_status():
            print("Waiting for application status.", file=self.output)
            sleep(10)

        print("Finished deployment.", file=self.output)

        return app

    def deploy(
        self,
        application_package: ApplicationPackage,
    ) -> Vespa:
        """
        Deploy the application package into a Vespa container.
        :param application_package: ApplicationPackage to be deployed.
        :return: a Vespa connection instance.
        """
        if not self.disk_folder:
            self.disk_folder = os.path.join(os.getcwd(), application_package.name)

        self.export_application_package(application_package=application_package)

        return self._execute_deployment(
            application_name=application_package.name,
            disk_folder=self.disk_folder,
            container_memory=self.container_memory,
            application_folder="application",
            application_package=application_package,
        )

    def deploy_from_disk(
        self,
        application_name: str,
        application_folder: Optional[str] = None,
    ) -> Vespa:
        """
        Deploy disk-based application package into a Vespa container.

        :param application_name: Name of the application.
        :param application_folder: Relative path to the folder inside `disk_folder` containing the application files.
            If None, we assume `disk_folder` to be the application folder.
        :return: a Vespa connection instance.
        """
        if not self.disk_folder:
            self.disk_folder = os.path.join(os.getcwd(), application_name)

        return self._execute_deployment(
            application_name=application_name,
            disk_folder=self.disk_folder,
            container_memory=self.container_memory,
            application_folder=application_folder,
        )

    def stop_services(self):
        """
        Stop Vespa services.

        :return: None
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

    def start_services(self):
        """
        Start Vespa services.

        :return: None
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
            while not app.get_application_status():
                print("Waiting for application status.", file=self.output)
                sleep(10)
            for line in start_services.output.decode("utf-8").split("\n"):
                print(line, file=self.output)
        else:
            raise RuntimeError("No container found")

    def restart_services(self):
        """
        Restart Vespa  services.

        :return: None
        """
        self.stop_services()
        self.start_services()

    @staticmethod
    def from_dict(mapping: Mapping) -> "VespaDocker":
        try:
            if mapping["container_id"] is not None:
                vespa_docker = VespaDocker.from_container_name_or_id(
                    name_or_id=mapping["container_id"]
                )
                return vespa_docker
            elif mapping["container_name"] is not None:
                vespa_docker = VespaDocker.from_container_name_or_id(
                    name_or_id=mapping["container_name"]
                )
                return vespa_docker
            else:
                print(
                    "Unable to instantiate VespaDocker from a running container. Starting new container."
                )
        except ValueError:
            print(
                "Unable to instantiate VespaDocker from a running container. Starting new container."
            )
        vespa_docker = VespaDocker(
            disk_folder=mapping["disk_folder"],
            port=mapping["port"],
            container_memory=mapping["container_memory"],
        )
        return vespa_docker

    @property
    def to_dict(self) -> Mapping:
        map = {
            "container_id": self.container_id,
            "container_name": self.container_name,
            "url": self.url,
            "port": self.local_port,
            "disk_folder": self.disk_folder,
            "container_memory": self.container_memory,
        }
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.container_id == other.container_id
            and self.container_name == other.container_name
            and self.url == other.url
            and self.local_port == other.local_port
            and self.disk_folder == other.disk_folder
            and self.container_memory == other.container_memory
        )

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5}, {6})".format(
            self.__class__.__name__,
            repr(self.disk_folder),
            repr(self.url),
            repr(self.local_port),
            repr(self.container_name),
            repr(self.container_id),
            repr(self.container_memory),
        )


class VespaCloud(object):
    def __init__(
        self,
        tenant: str,
        application: str,
        application_package: ApplicationPackage,
        key_location: Optional[str] = None,
        key_content: Optional[str] = None,
        output_file: IO = sys.stdout,
    ) -> None:
        """
        Deploy application to the Vespa Cloud (cloud.vespa.ai)

        :param tenant: Tenant name registered in the Vespa Cloud.
        :param application: Application name registered in the Vespa Cloud.
        :param application_package: ApplicationPackage to be deployed.
        :param key_location: Location of the private key used for signing HTTP requests to the Vespa Cloud.
        :param key_content: Content of the private key used for signing HTTP requests to the Vespa Cloud. Use only when
            key file is not available.
        :param output_file: Output file to write output messages.
        """
        self.tenant = tenant
        self.application = application
        self.application_package = application_package
        self.api_key = self._read_private_key(key_location, key_content)
        self.api_public_key_bytes = standard_b64encode(
            self.api_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        self.data_key, self.data_certificate = self._create_certificate_pair()
        self.private_cert_file_name = "private_cert.txt"
        self.connection = http.client.HTTPSConnection(
            "api.vespa-external.aws.oath.cloud", 4443
        )
        self.output = output_file

    @staticmethod
    def _read_private_key(
        key_location: Optional[str] = None, key_content: Optional[str] = None
    ) -> ec.EllipticCurvePrivateKey:

        if key_content:
            key_content = bytes(key_content, "ascii")
        elif key_location:
            with open(key_location, "rb") as key_data:
                key_content = key_data.read()
        else:
            raise ValueError("Provide either key_content or key_location.")

        key = serialization.load_pem_private_key(key_content, None, default_backend())
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise TypeError("Key must be an elliptic curve private key")
        return key

    def _write_private_key_and_cert(
        self, key: ec.EllipticCurvePrivateKey, cert: x509.Certificate, disk_folder: str
    ) -> None:
        cert_file = os.path.join(disk_folder, self.private_cert_file_name)
        with open(cert_file, "w+") as file:
            file.write(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption(),
                ).decode("UTF-8")
            )
            file.write(cert.public_bytes(serialization.Encoding.PEM).decode("UTF-8"))

    @staticmethod
    def _create_certificate_pair() -> (ec.EllipticCurvePrivateKey, x509.Certificate):
        key = ec.generate_private_key(ec.SECP384R1, default_backend())
        name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, u"localhost")])
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow() - timedelta(minutes=1))
            .not_valid_after(datetime.utcnow() + timedelta(days=7))
            .public_key(key.public_key())
            .sign(key, hashes.SHA256(), default_backend())
        )
        return key, certificate

    def _request(
        self, method: str, path: str, body: BytesIO = BytesIO(), headers={}
    ) -> dict:
        digest = hashes.Hash(hashes.SHA256(), default_backend())
        body.seek(0)
        digest.update(body.read())
        content_hash = standard_b64encode(digest.finalize()).decode("UTF-8")
        timestamp = (
            datetime.utcnow().isoformat() + "Z"
        )  # Java's Instant.parse requires the neutral time zone appended
        url = "https://" + self.connection.host + ":" + str(self.connection.port) + path

        canonical_message = method + "\n" + url + "\n" + timestamp + "\n" + content_hash
        signature = self.api_key.sign(
            canonical_message.encode("UTF-8"), ec.ECDSA(hashes.SHA256())
        )

        headers = {
            "X-Timestamp": timestamp,
            "X-Content-Hash": content_hash,
            "X-Key-Id": self.tenant + ":" + self.application + ":" + "default",
            "X-Key": self.api_public_key_bytes,
            "X-Authorization": standard_b64encode(signature),
            **headers,
        }

        body.seek(0)
        self.connection.request(method, path, body, headers)
        with self.connection.getresponse() as response:
            parsed = json.load(response)
            if response.status != 200:
                raise RuntimeError(
                    "Status code "
                    + str(response.status)
                    + " doing "
                    + method
                    + " at "
                    + url
                    + ":\n"
                    + parsed["message"]
                )
            return parsed

    def _get_dev_region(self) -> str:
        return self._request("GET", "/zone/v1/environment/dev/default")["name"]

    def _get_endpoint(self, instance: str, region: str) -> str:
        endpoints = self._request(
            "GET",
            "/application/v4/tenant/{}/application/{}/instance/{}/environment/dev/region/{}".format(
                self.tenant, self.application, instance, region
            ),
        )["endpoints"]
        container_url = [
            endpoint["url"]
            for endpoint in endpoints
            if endpoint["cluster"]
            == "{}_container".format(self.application_package.name)
        ]
        if not container_url:
            raise RuntimeError("No endpoints found for container 'test_app_container'")
        return container_url[0]

    def _to_application_zip(self, disk_folder) -> BytesIO:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "a") as zip_archive:

            for schema in self.application_package.schemas:
                zip_archive.writestr(
                    "application/schemas/{}.sd".format(schema.name),
                    schema.schema_to_text,
                )
                for model in schema.models:
                    zip_archive.write(
                        model.model_file_path,
                        os.path.join("application/files", model.model_file_name),
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
                        "application/models/{}.onnx".format(model_id),
                    )
                    os.remove(temp_model_file)

            if self.application_package.query_profile:
                zip_archive.writestr(
                    "application/search/query-profiles/default.xml",
                    self.application_package.query_profile_to_text,
                )
                zip_archive.writestr(
                    "application/search/query-profiles/types/root.xml",
                    self.application_package.query_profile_type_to_text,
                )
            zip_archive.writestr(
                "application/services.xml", self.application_package.services_to_text
            )
            zip_archive.writestr(
                "application/security/clients.pem",
                self.data_certificate.public_bytes(serialization.Encoding.PEM),
            )

        return buffer

    def _start_deployment(self, instance: str, job: str, disk_folder: str) -> int:
        deploy_path = (
            "/application/v4/tenant/{}/application/{}/instance/{}/deploy/{}".format(
                self.tenant, self.application, instance, job
            )
        )

        Path(disk_folder).mkdir(parents=True, exist_ok=True)

        application_zip_bytes = self._to_application_zip(disk_folder=disk_folder)

        self._write_private_key_and_cert(
            self.data_key, self.data_certificate, disk_folder
        )

        response = self._request(
            "POST",
            deploy_path,
            application_zip_bytes,
            {"Content-Type": "application/zip"},
        )
        print(response["message"], file=self.output)
        return response["run"]

    def _get_deployment_status(
        self, instance: str, job: str, run: int, last: int
    ) -> (str, int):

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
            elif status in fail_status_message.keys():
                raise RuntimeError(fail_status_message[status])
            else:
                raise RuntimeError("Unexpected status: {}".format(status))

    def _follow_deployment(self, instance: str, job: str, run: int) -> None:
        last = -1
        while True:
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

    def _print_log_entry(self, step: str, entry: dict):
        timestamp = strftime("%H:%M:%S", gmtime(entry["at"] / 1e3))
        message = entry["message"].replace("\n", "\n" + " " * 23)
        if step != "copyVespaLogs" or entry["type"] == "error":
            print(
                "{:<7} [{}]  {}".format(entry["type"].upper(), timestamp, message),
                file=self.output,
            )

    def deploy(self, instance: str, disk_folder: Optional[str] = None) -> Vespa:
        """
        Deploy the given application package as the given instance in the Vespa Cloud dev environment.

        :param instance: Name of this instance of the application, in the Vespa Cloud.
        :param disk_folder: Disk folder to save the required Vespa config files. Default to application name
            folder within user's current working directory.

        :return: a Vespa connection instance.
        """

        if not disk_folder:
            disk_folder = os.path.join(os.getcwd(), self.application_package.name)

        region = self._get_dev_region()
        job = "dev-" + region
        run = self._start_deployment(instance, job, disk_folder)
        self._follow_deployment(instance, job, run)
        endpoint_url = self._get_endpoint(instance=instance, region=region)
        print("Finished deployment.", file=self.output)
        return Vespa(
            url=endpoint_url,
            cert=os.path.join(disk_folder, self.private_cert_file_name),
            application_package=self.application_package
        )

    def delete(self, instance: str):
        """
        Delete the specified instance from the dev environment in the Vespa Cloud.
        :param instance: Name of the instance to delete.
        :return:
        """
        print(
            self._request(
                "DELETE",
                "/application/v4/tenant/{}/application/{}/instance/{}/environment/dev/region/{}".format(
                    self.tenant, self.application, instance, self._get_dev_region()
                ),
            )["message"],
            file=self.output,
        )
        print(
            self._request(
                "DELETE",
                "/application/v4/tenant/{}/application/{}/instance/{}".format(
                    self.tenant, self.application, instance
                ),
            )["message"],
            file=self.output,
        )

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
