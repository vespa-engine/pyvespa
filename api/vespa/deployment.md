## `vespa.deployment`

### `VespaDeployment`

#### `read_app_package_from_disk(application_root)`

Reads the contents of an application package on disk into a zip file.

Parameters:

| Name               | Type  | Description                                    | Default    |
| ------------------ | ----- | ---------------------------------------------- | ---------- |
| `application_root` | `str` | The directory root of the application package. | *required* |

Returns:

| Name    | Type    | Description                     |
| ------- | ------- | ------------------------------- |
| `bytes` | `bytes` | The zipped application package. |

### `VespaDocker(url='http://localhost', port=8080, container_memory=0, output_file=sys.stdout, container=None, container_image='vespaengine/vespa', volumes=None, cfgsrv_port=19071, debug_port=5005)`

Bases: `VespaDeployment`

Manage Docker deployments.

Make sure to start the Docker daemon before instantiating this class.

Example usage

```python
from vespa.deployment import VespaDocker

vespa_docker = VespaDocker(port=8080)
# or initialize from a running container:
vespa_docker = VespaDocker('http://localhost', 8080, None, None, 4294967296, 'vespaengine/vespa')
```

**Note**:

It is **NOT** possible to refer to Volume Mounts in your Application Package. This means that for example .onnx-model files that are part of the Application Package **must** be on your host machine, so that it can be uploaded as part of the Application Package to the Vespa container.

Parameters:

| Name               | Type          | Description                                                                                                                                                                                                                                      | Default               |
| ------------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------- |
| `port`             | `int`         | The port for the container. Default is 8080.                                                                                                                                                                                                     | `8080`                |
| `cfgsrv_port`      | `int`         | The Vespa Config Server port. Default is 19071.                                                                                                                                                                                                  | `19071`               |
| `debug_port`       | `int`         | The port to connect to for debugging the Vespa container. Default is 5005.                                                                                                                                                                       | `5005`                |
| `output_file`      | `str`         | The file to write output messages to.                                                                                                                                                                                                            | `stdout`              |
| `container_memory` | `int`         | Container maximum memory usage limit, in bytes. Default is 0 (unlimited). Processes will be killed if memory usage exceeds this limit. Set to 0 for no limit, or specify a limit like 8 * 1024\*\*3 for 8GB. Vespa requires at least 4GB to run. | `0`                   |
| `container`        | `str`         | Used when instantiating VespaDocker from a running container.                                                                                                                                                                                    | `None`                |
| `volumes`          | `list of str` | A list of volume mount strings, such as ['/home/user1/:/mnt/vol2', '/var/www:/mnt/vol1']. The Application Package cannot reference volume mounts.                                                                                                | `None`                |
| `container_image`  | `str`         | The Docker container image to use.                                                                                                                                                                                                               | `'vespaengine/vespa'` |
| `url`              | `str`         | The URL to connect to the Vespa instance. Default is "http://localhost".                                                                                                                                                                         | `'http://localhost'`  |

#### `from_container_name_or_id(name_or_id, output_file=sys.stdout)`

Instantiate VespaDocker from a running container.

Parameters:

| Name          | Type  | Description                              | Default    |
| ------------- | ----- | ---------------------------------------- | ---------- |
| `name_or_id`  | `str` | The name or id of the running container. | *required* |
| `output_file` | `str` | The file to write output messages to.    | `stdout`   |

Raises:

| Type         | Description                              |
| ------------ | ---------------------------------------- |
| `ValueError` | If the specified container is not found. |

Returns:

| Name          | Type          | Description                                                       |
| ------------- | ------------- | ----------------------------------------------------------------- |
| `VespaDocker` | `VespaDocker` | An instance of VespaDocker associated with the running container. |

#### `deploy(application_package, max_wait_configserver=60, max_wait_deployment=300, max_wait_docker=300, debug=False)`

Deploy the application package into a Vespa container.

Parameters:

| Name                    | Type                 | Description                                                         | Default    |
| ----------------------- | -------------------- | ------------------------------------------------------------------- | ---------- |
| `application_package`   | `ApplicationPackage` | The application package to be deployed.                             | *required* |
| `max_wait_configserver` | `int`                | Maximum seconds to wait for the config server to start.             | `60`       |
| `max_wait_deployment`   | `int`                | Maximum seconds to wait for the deployment to complete.             | `300`      |
| `max_wait_docker`       | `int`                | Maximum seconds to wait for the Docker container to start.          | `300`      |
| `debug`                 | `bool`               | If True, adds the configured debug_port to the Docker port mapping. | `False`    |

Returns:

| Name              | Type    | Description                                                  |
| ----------------- | ------- | ------------------------------------------------------------ |
| `VespaConnection` | `Vespa` | A Vespa connection instance once the deployment is complete. |

#### `deploy_from_disk(application_name, application_root, max_wait_configserver=60, max_wait_application=300, docker_timeout=300, debug=False)`

Deploy from a directory tree.

This method is used when making changes to application package files that are not supported by pyvespa. This is why this method is not found in the ApplicationPackage class.

Parameters:

| Name               | Type   | Description                                                         | Default    |
| ------------------ | ------ | ------------------------------------------------------------------- | ---------- |
| `application_name` | `str`  | The name of the application package.                                | *required* |
| `application_root` | `str`  | The root directory of the application package.                      | *required* |
| `debug`            | `bool` | If True, adds the configured debug_port to the Docker port mapping. | `False`    |

Returns:

| Name              | Type    | Description                                                  |
| ----------------- | ------- | ------------------------------------------------------------ |
| `VespaConnection` | `Vespa` | A Vespa connection instance once the deployment is complete. |

#### `wait_for_config_server_start(max_wait=300)`

Waits for the Config Server to start inside the Docker image.

Parameters:

| Name       | Type  | Description                                                                             | Default |
| ---------- | ----- | --------------------------------------------------------------------------------------- | ------- |
| `max_wait` | `int` | The maximum number of seconds to wait for the application endpoint to become available. | `300`   |

Raises:

| Type           | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `RuntimeError` | If the config server does not start within the specified max_wait time. |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `start_services(max_wait=120)`

Start Vespa services inside the Docker image, first waiting for the Config Server, then for other services.

Parameters:

| Name       | Type  | Description                                                                             | Default |
| ---------- | ----- | --------------------------------------------------------------------------------------- | ------- |
| `max_wait` | `int` | The maximum number of seconds to wait for the application endpoint to become available. | `120`   |

Raises:

| Type           | Description                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------- |
| `RuntimeError` | If a container has not been set or the services fail to start within the specified max_wait time. |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `stop_services()`

Stop Vespa services inside the Docker image, first stopping the services, then stopping the Config Server.

Raises:

| Type           | Description                                                                     |
| -------------- | ------------------------------------------------------------------------------- |
| `RuntimeError` | If a container has not been set or an error occurs while stopping the services. |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `restart_services()`

Restart Vespa services inside the Docker image. This is equivalent to calling `self.stop_services()` followed by `self.start_services()`.

Raises:

| Type           | Description                                                                    |
| -------------- | ------------------------------------------------------------------------------ |
| `RuntimeError` | If a container has not been set or an error occurs during the restart process. |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

### `VespaCloud(tenant, application, application_package=None, key_location=None, key_content=None, auth_client_token_id=None, output_file=sys.stdout, application_root=None, cluster=None, instance='default')`

Bases: `VespaDeployment`

Deploy an application to the Vespa Cloud (cloud.vespa.ai).

There are several ways to initialize VespaCloud:

- Application source: From a Python-defined application package or from the application_root folder.
- Control plane access: Using an API key (must be added to Vespa Cloud Console) or an access token, obtained by interactive login.
- Data plane access: mTLS is used by default, but Vespa applications can also be configured to use token-based authentication (token must be added to Vespa Cloud Console, and the corresponding auth_token_id must be provided).

Example usage

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

Parameters:

| Name                   | Type                 | Description                                                                                                                                                                                    | Default     |
| ---------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `tenant`               | `str`                | Tenant name registered in the Vespa Cloud.                                                                                                                                                     | *required*  |
| `application`          | `str`                | Application name in the Vespa Cloud.                                                                                                                                                           | *required*  |
| `application_package`  | `ApplicationPackage` | Application package to be deployed. Either this or application_root must be set.                                                                                                               | `None`      |
| `key_location`         | `str`                | Location of the control plane key used for signing HTTP requests to the Vespa Cloud.                                                                                                           | `None`      |
| `key_content`          | `str`                | Content of the control plane key used for signing HTTP requests to the Vespa Cloud. Use only when the key file is not available.                                                               | `None`      |
| `auth_client_token_id` | `str`                | Token-based data plane authentication. This token name must be configured in the Vespa Cloud Console. It configures Vespa's services.xml, and the token must have read and write permissions.  | `None`      |
| `output_file`          | `str`                | Output file to write output messages. Default is sys.stdout.                                                                                                                                   | `stdout`    |
| `application_root`     | `str`                | Directory for the application root (location of services.xml, models/, schemas/, etc.). If the application is packaged with Maven, use the generated <myapp>/target/application directory.     | `None`      |
| `cluster`              | `str`                | Name of the cluster to target when retrieving endpoints. This affects which endpoints are used for initializing the :class:Vespa instance in VespaCloud.get_application and VespaCloud.deploy. | `None`      |
| `instance`             | `str`                | Name of the application instance. Default is "default".                                                                                                                                        | `'default'` |

Raises:

| Type           | Description          |
| -------------- | -------------------- |
| `RuntimeError` | If deployment fails. |

Returns:

| Name    | Type   | Description                                                                |
| ------- | ------ | -------------------------------------------------------------------------- |
| `Vespa` | `None` | A Vespa connection instance for interacting with the deployed application. |

#### `deploy(instance='default', disk_folder=None, version=None, max_wait=1800, environment='dev', region=None)`

Deploy the given application package as the given instance in the Vespa Cloud dev or perf environment.

Parameters:

| Name          | Type                     | Description                                                                                                                                                                                       | Default     |
| ------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `instance`    | `str`                    | Name of this instance of the application in the Vespa Cloud.                                                                                                                                      | `'default'` |
| `disk_folder` | `str`                    | Disk folder to save the required Vespa config files. Defaults to the application name folder within the user's current working directory.                                                         | `None`      |
| `version`     | `str`                    | Vespa version to use for deployment. Defaults to None, meaning the latest version. Should only be set based on instructions from the Vespa team. Must be a valid Vespa version, e.g., "8.435.13". | `None`      |
| `max_wait`    | `int`                    | Seconds to wait for the deployment to complete.                                                                                                                                                   | `1800`      |
| `environment` | `Literal['dev', 'perf']` | Environment to deploy to. Default is "dev".                                                                                                                                                       | `'dev'`     |
| `region`      | `str`                    | Dev region to deploy to. Valid regions: "aws-us-east-1c" (default), "aws-euw1-az1", "azure-eastus-az1", "gcp-us-central1-f". Only used when environment is "dev".                                 | `None`      |

Returns:

| Name    | Type    | Description                                                                                                                                                        |
| ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Vespa` | `Vespa` | A Vespa connection instance. This instance connects to the mTLS endpoint. To connect to the token endpoint, use VespaCloud.get_application(endpoint_type="token"). |

Raises:

| Type           | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `RuntimeError` | If deployment fails or if there are issues with the deployment process. |
| `ValueError`   | If an invalid dev region is provided.                                   |

#### `deploy_to_prod(instance='default', application_root=None, source_url='')`

Deploy the given application package as the given instance in the Vespa Cloud prod environment. NB! This feature is experimental and may fail in unexpected ways. Expect better support in future releases.

If submitting an application that is not yet packaged, tests should be located in /tests. If submitting an application packaged with maven, application_root should refer to the generated /target/application directory.

Parameters:

| Name               | Type  | Description                                                                                                                                                                                                                                                                                                                                                       | Default     |
| ------------------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `instance`         | `str` | Name of this instance of the application in the Vespa Cloud.                                                                                                                                                                                                                                                                                                      | `'default'` |
| `application_root` | `str` | Path to either save the required Vespa config files (if initialized with application_package) or read them from (if initialized with application_root).                                                                                                                                                                                                           | `None`      |
| `source_url`       | `str` | Optional source URL (including commit hash) for the deployment. This is a URL to the source code repository, e.g., GitHub, that is used to build the application package. Example: https://github.com/vespa-cloud/vector-search/commit/474d7771bd938d35dc5dcfd407c21c019d15df3c. The source URL will show up in the Vespa Cloud Console next to the build number. | `''`        |

Raises:

| Type           | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `RuntimeError` | If deployment fails or if there are issues with the deployment process. |

#### `get_application(instance='default', environment='dev', endpoint_type='mtls', vespa_cloud_secret_token=None, region=None, max_wait=60)`

Get a connection to the Vespa application instance. Will only work if the application is already deployed.

Example usage

```python
vespa_cloud = VespaCloud(...)
app: Vespa = vespa_cloud.get_application()
# Feed, query, visit, etc.
```

Parameters:

| Name                       | Type  | Description                                                                                                                            | Default     |
| -------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `instance`                 | `str` | Name of this instance of the application in the Vespa Cloud. Default is "default".                                                     | `'default'` |
| `environment`              | `str` | Environment of the application. Default is "dev". Options are "dev", "perf", or "prod".                                                | `'dev'`     |
| `endpoint_type`            | `str` | Type of endpoint to connect to. Default is "mtls". Options are "mtls" or "token".                                                      | `'mtls'`    |
| `vespa_cloud_secret_token` | `str` | Vespa Cloud Secret Token. Only required if endpoint_type is "token".                                                                   | `None`      |
| `region`                   | `str` | Region of the application in Vespa Cloud, e.g., "aws-us-east-1c". If not provided, the first region from the environment will be used. | `None`      |
| `max_wait`                 | `int` | Seconds to wait for the application to be up. Default is 60 seconds.                                                                   | `60`        |

Returns:

| Name    | Type    | Description                 |
| ------- | ------- | --------------------------- |
| `Vespa` | `Vespa` | Vespa application instance. |

Raises:

| Type           | Description                                                                           |
| -------------- | ------------------------------------------------------------------------------------- |
| `RuntimeError` | If the application is not yet deployed or there are issues retrieving the connection. |

#### `check_production_build_status(build_no, quiet=False)`

Check the status of a production build. Useful for example in CI/CD pipelines to check when a build has converged.

Example usage

```python
vespa_cloud = VespaCloud(...)
build_no = vespa_cloud.deploy_to_prod()
status = vespa_cloud.check_production_build_status(build_no)
# The response contains:
# - "deployed" (bool): True if the build has converged everywhere.
# - "status" (str): "deploying" or "done".
# - "hasFailed" (bool): True if any job for this build has ever failed.
#       Once true, it stays true even if the system retries with a new run.
# - "skipReason" (str, optional): Why the build was skipped, e.g. "no-changes" or "cancelled".
# - "jobs" (list): Per-job deployment details, each with "jobName", "runStatus",
#       "runId", and "instance". The list grows as jobs are triggered.
#
# Each job shows the most recent run's status for this build.
#
# Example: early in deployment (only tests triggered so far):
#    {"deployed": False, "status": "deploying", "hasFailed": False,
#     "jobs": [{"jobName": "system-test", "runStatus": "running"},
#              {"jobName": "staging-test", "runStatus": "running"}]}
#
# Example: fully deployed:
#    {"deployed": True, "status": "done", "hasFailed": False,
#     "jobs": [{"jobName": "system-test", "runStatus": "success"},
#              {"jobName": "staging-test", "runStatus": "success"},
#              {"jobName": "production-us-east-3", "runStatus": "success"}]}
#
# Example: a job failed (system retries, but hasFailed stays true):
#    {"deployed": False, "status": "deploying", "hasFailed": True,
#     "jobs": [{"jobName": "system-test", "runStatus": "success"},
#              {"jobName": "staging-test", "runStatus": "installationFailed"}]}
#
# Example: skipped before any jobs triggered (no changes):
#    {"deployed": False, "status": "done", "hasFailed": False,
#     "skipReason": "no-changes", "jobs": []}
#
# Example: cancelled after some jobs ran:
#    {"deployed": False, "status": "done", "hasFailed": True,
#     "skipReason": "cancelled",
#     "jobs": [{"jobName": "system-test", "runStatus": "success"},
#              {"jobName": "staging-test", "runStatus": "running"}]}
```

Parameters:

| Name       | Type   | Description                                       | Default    |
| ---------- | ------ | ------------------------------------------------- | ---------- |
| `build_no` | `int`  | The build number to check.                        | *required* |
| `quiet`    | `bool` | If True, suppress status print. Default is False. | `False`    |

Returns:

| Name   | Type   | Description                                                                             |
| ------ | ------ | --------------------------------------------------------------------------------------- |
| `dict` | `dict` | The build status response from the API. See example responses above for the full shape. |

Raises:

| Type           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `RuntimeError` | If there are issues with retrieving the status of the build. |

#### `wait_for_prod_deployment(build_no=None, max_wait=3600, poll_interval=5)`

Wait for a production deployment to finish by polling build status. Prints per-job status changes as they happen (only prints when a job's status changes).

Example usage

```python
vespa_cloud = VespaCloud(...)
build_no = vespa_cloud.deploy_to_prod()
success = vespa_cloud.wait_for_prod_deployment(build_no, max_wait=3600, poll_interval=5)
print(success)
# Output: True
```

Parameters:

| Name            | Type  | Description                                                                   | Default |
| --------------- | ----- | ----------------------------------------------------------------------------- | ------- |
| `build_no`      | `int` | The build number to check.                                                    | `None`  |
| `max_wait`      | `int` | Maximum time to wait for the deployment in seconds. Default is 3600 (1 hour). | `3600`  |
| `poll_interval` | `int` | Polling interval in seconds. Default is 5 seconds.                            | `5`     |

Returns:

| Name   | Type   | Description                                                                                                                       |
| ------ | ------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `bool` | `bool` | True if the build was deployed to all production zones, False if it completed without deploying (e.g. skipped due to no changes). |

Raises:

| Type           | Description                                                                                                                            |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `RuntimeError` | If any job for this build has failed. The deployment system may continue retrying, but this method exits immediately on first failure. |
| `TimeoutError` | If the deployment did not finish within max_wait seconds.                                                                              |

#### `deploy_from_disk(instance, application_root, max_wait=300, version=None, environment='dev', region=None)`

Deploy to the development or performance environment from a directory tree. This method is used when making changes to application package files that are not supported by pyvespa. Note: Requires a certificate and key to be generated using 'vespa auth cert'.

Example usage

```python
vespa_cloud = VespaCloud(...)
vespa_cloud.deploy_from_disk(
    instance="my-instance",
    application_root="/path/to/application",
    max_wait=3600,
    version="8.435.13"
)
```

Parameters:

| Name               | Type  | Description                                                                                                                                                       | Default    |
| ------------------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `instance`         | `str` | The name of the instance where the application will be run.                                                                                                       | *required* |
| `application_root` | `str` | The root directory of the application package.                                                                                                                    | *required* |
| `max_wait`         | `int` | The maximum number of seconds to wait for the deployment. Default is 3600 (1 hour).                                                                               | `300`      |
| `version`          | `str` | The Vespa version to use for the deployment. Default is None, which means the latest version. It must be a valid Vespa version (e.g., "8.435.13").                | `None`     |
| `environment`      | `str` | Environment to deploy to. Default is "dev". Options are "dev" or "perf".                                                                                          | `'dev'`    |
| `region`           | `str` | Dev region to deploy to. Valid regions: "aws-us-east-1c" (default), "aws-euw1-az1", "azure-eastus-az1", "gcp-us-central1-f". Only used when environment is "dev". | `None`     |

Returns:

| Name    | Type    | Description                                                                                                                                               |
| ------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Vespa` | `Vespa` | A Vespa connection instance. This connects to the mtls endpoint. To connect to the token endpoint, use VespaCloud.get_application(endpoint_type="token"). |

#### `delete(instance='default', environment='dev', region=None)`

Delete the specified instance from the development environment in the Vespa Cloud.

To delete a production instance, you must submit a new deployment with `deployment-removal` added to the 'validation-overrides.xml'. See <https://cloud.vespa.ai/en/deleting-applications> for more details.

Example usage

```python
vespa_cloud = VespaCloud(...)
vespa_cloud.delete_instance(instance="my-instance")
```

Parameters:

| Name          | Type  | Description                                                                                                                                                         | Default     |
| ------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `instance`    | `str` | The name of the instance to delete.                                                                                                                                 | `'default'` |
| `environment` | `str` | The environment from which to delete the instance. Must be "dev" or "perf".                                                                                         | `'dev'`     |
| `region`      | `str` | Dev region to delete from. Valid regions: "aws-us-east-1c" (default), "aws-euw1-az1", "azure-eastus-az1", "gcp-us-central1-f". Only used when environment is "dev". | `None`      |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `get_all_endpoints(instance='default', region=None, environment='dev')`

Get all endpoints for the application instance.

Parameters:

| Name          | Type  | Description                         | Default     |
| ------------- | ----- | ----------------------------------- | ----------- |
| `instance`    | `str` | Application instance name.          | `'default'` |
| `region`      | `str` | Region name, e.g. 'aws-us-east-1c'. | `None`      |
| `environment` | `str` | Environment (dev/perf/prod).        | `'dev'`     |

Returns:

| Name   | Type                   | Description        |
| ------ | ---------------------- | ------------------ |
| `list` | `List[Dict[str, str]]` | List of endpoints. |

#### `get_private_services(instance='default', region=None, environment='dev')`

Get private services for the application instance.

Parameters:

| Name          | Type  | Description                         | Default     |
| ------------- | ----- | ----------------------------------- | ----------- |
| `instance`    | `str` | Application instance name.          | `'default'` |
| `region`      | `str` | Region name, e.g. 'aws-us-east-1c'. | `None`      |
| `environment` | `str` | Environment (dev/perf/prod).        | `'dev'`     |

Returns:

| Name   | Type   | Description                |
| ------ | ------ | -------------------------- |
| `dict` | `dict` | Private services response. |

warning:: This method is experimental and may change.

#### `get_app_package_contents(instance='default', region=None, environment='dev')`

Get all endpoints for the application package content in the specified region and environment.

Parameters:

| Name          | Type  | Description                                                                               | Default     |
| ------------- | ----- | ----------------------------------------------------------------------------------------- | ----------- |
| `instance`    | `str` | Application instance name.                                                                | `'default'` |
| `region`      | `str` | Region name, e.g. 'aws-us-east-1c'. If None, uses the default region for the environment. | `None`      |
| `environment` | `str` | Environment (dev/perf/prod). Default is 'dev'.                                            | `'dev'`     |

Returns:

| Name   | Type        | Description                                     |
| ------ | ----------- | ----------------------------------------------- |
| `list` | `List[str]` | List of endpoints for the application instance. |

#### `get_schemas(instance='default', region=None, environment='dev')`

Get all schemas for the application instance in the specified environment and region.

Parameters:

| Name          | Type  | Description                                                                               | Default     |
| ------------- | ----- | ----------------------------------------------------------------------------------------- | ----------- |
| `instance`    | `str` | Application instance name.                                                                | `'default'` |
| `region`      | `str` | Region name, e.g. 'aws-us-east-1c'. If None, uses the default region for the environment. | `None`      |
| `environment` | `str` | Environment (dev/perf/prod). Default is 'dev'.                                            | `'dev'`     |

Returns:

| Name   | Type             | Description                                              |
| ------ | ---------------- | -------------------------------------------------------- |
| `dict` | `Dict[str, str]` | Dictionary with schema name as key and content as value. |

#### `download_app_package_content(destination_path, instance='default', region=None, environment='dev')`

Download the application package content to a specified destination path.

Parameters:

| Name               | Type  | Description                                                                               | Default     |
| ------------------ | ----- | ----------------------------------------------------------------------------------------- | ----------- |
| `destination_path` | `str` | The path where the application package content will be downloaded.                        | *required*  |
| `instance`         | `str` | Application instance name.                                                                | `'default'` |
| `region`           | `str` | Region name, e.g. 'aws-us-east-1c'. If None, uses the default region for the environment. | `None`      |
| `environment`      | `str` | Environment (dev/perf/prod). Default is 'dev'.                                            | `'dev'`     |

Returns:

| Type   | Description |
| ------ | ----------- |
| `None` | None        |

#### `get_endpoint_auth_method(url, instance='default', region=None, environment='dev')`

Get the authentication method for the given endpoint URL.

Parameters:

| Name          | Type  | Description                         | Default     |
| ------------- | ----- | ----------------------------------- | ----------- |
| `url`         | `str` | The endpoint URL.                   | *required*  |
| `instance`    | `str` | Application instance name.          | `'default'` |
| `region`      | `str` | Region name, e.g. 'aws-us-east-1c'. | `None`      |
| `environment` | `str` | Environment (dev/perf/prod).        | `'dev'`     |

Returns:

| Name  | Type  | Description                                    |
| ----- | ----- | ---------------------------------------------- |
| `str` | `str` | The authentication method ('mtls' or 'token'). |

#### `get_endpoint(auth_method, instance='default', region=None, environment='dev', cluster=None)`

Get the endpoint URL for the application.

Tip: See the 'endpoint'-tab in Vespa Cloud Console for available endpoints.

Parameters:

| Name          | Type  | Description                                                                             | Default     |
| ------------- | ----- | --------------------------------------------------------------------------------------- | ----------- |
| `auth_method` | `str` | Authentication method. Options are 'mtls' or 'token'.                                   | *required*  |
| `instance`    | `str` | Application instance name.                                                              | `'default'` |
| `region`      | `str` | Region name, e.g. 'aws-us-east-1c'.                                                     | `None`      |
| `environment` | `str` | Environment (dev/perf/prod).                                                            | `'dev'`     |
| `cluster`     | `str` | Specific cluster to get the endpoint for. If None, uses the instance's default cluster. | `None`      |

Returns:

| Name  | Type  | Description       |
| ----- | ----- | ----------------- |
| `str` | `str` | The endpoint URL. |

#### `get_mtls_endpoint(instance='default', region=None, environment='dev', cluster=None)`

Get the endpoint URL of a mTLS endpoint for the application. Will return the first mTLS endpoint found if multiple exist. Use `VespaCloud.get_all_endpoints` to get all endpoints.

Tip: See the 'endpoint'-tab in Vespa Cloud Console for available endpoints.

Parameters:

| Name          | Type  | Description                                                                             | Default     |
| ------------- | ----- | --------------------------------------------------------------------------------------- | ----------- |
| `instance`    | `str` | Application instance name.                                                              | `'default'` |
| `region`      | `str` | Region name.                                                                            | `None`      |
| `environment` | `str` | Environment (dev/perf/prod).                                                            | `'dev'`     |
| `cluster`     | `str` | Specific cluster to get the endpoint for. If None, uses the instance's default cluster. | `None`      |

Returns:

| Name  | Type  | Description       |
| ----- | ----- | ----------------- |
| `str` | `str` | The endpoint URL. |

#### `get_token_endpoint(instance='default', region=None, environment='dev', cluster=None)`

Get the endpoint URL of a token endpoint for the application. Will return the first token endpoint found if multiple exist. Use `VespaCloud.get_all_endpoints` to get all endpoints.

Tip: See the 'endpoint'-tab in Vespa Cloud Console for available endpoints.

Parameters:

| Name          | Type  | Description                                                                             | Default     |
| ------------- | ----- | --------------------------------------------------------------------------------------- | ----------- |
| `instance`    | `str` | Application instance name.                                                              | `'default'` |
| `region`      | `str` | Region name.                                                                            | `None`      |
| `environment` | `str` | Environment (dev/perf/prod).                                                            | `'dev'`     |
| `cluster`     | `str` | Specific cluster to get the endpoint for. If None, uses the instance's default cluster. | `None`      |

Returns:

| Name  | Type  | Description       |
| ----- | ----- | ----------------- |
| `str` | `str` | The endpoint URL. |
