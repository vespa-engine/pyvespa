# Authenticating to Vespa Cloud[¶](#authenticating-to-vespa-cloud)

Security is a top priority for the Vespa Team. We understand that as a newcomer to Vespa, the different authentication methods may not always be immediately clear.

This notebook is intended to provide some clarity on the different authentication methods needed when interacting with Vespa Cloud for different purposes.

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

**Pre-requisite**: Create a tenant at [cloud.vespa.ai](https://cloud.vespa.ai/), save the tenant name.

## Install[¶](#install)

Install [pyvespa](https://pyvespa.readthedocs.io/) >= 0.45 and the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html).

In \[1\]:

Copied!

```
!pip3 install pyvespa vespacli
```

!pip3 install pyvespa vespacli

For background context, it is useful to read the [Vespa Cloud Security Guide](https://cloud.vespa.ai/en/security/guide).

## Control-plane vs Data-plane[¶](#control-plane-vs-data-plane)

This may be self-explanatory for some, but it is worth mentioning that Vespa Cloud has two main components: the control-plane and the data-plane, which provide access to different functionalities.

|                                                                                                            | Control-plane | Data-plane | Comments                                                                                   |
| ---------------------------------------------------------------------------------------------------------- | ------------- | ---------- | ------------------------------------------------------------------------------------------ |
| Deploy application                                                                                         | ✅            | ❌         |                                                                                            |
| Modify application (re-deploy)                                                                             | ✅            | ❌         |                                                                                            |
| Add or modify data-plane certs or token(s)                                                                 | ✅            | ❌         |                                                                                            |
| Feed data                                                                                                  | ❌            | ✅         |                                                                                            |
| Query data                                                                                                 | ❌            | ✅         |                                                                                            |
| Delete data                                                                                                | ❌            | ✅         |                                                                                            |
| [Visiting](https://docs.vespa.ai/en/visiting.html)                                                         | ❌            | ✅         |                                                                                            |
| [Monitoring](https://cloud.vespa.ai/en/monitoring)                                                         | ❌            | ✅         |                                                                                            |
| Get application package                                                                                    | ✅            | ❌         |                                                                                            |
| [vespa auth login](https://docs.vespa.ai/en/clients/vespa-cli.html)                                        | ✅            | ❌         | Interactive control-plane login in browser                                                 |
| [vespa auth api-key](https://docs.vespa.ai/en/clients/vespa-cli.html)                                      | ✅            | ❌         | Headless control-plane authentication with an API key generated in the Vespa Cloud console |
| [vespa auth cert](https://docs.vespa.ai/en/clients/vespa-cli.html)                                         | ❌            | ✅         | Used to generate a certificate for a data-plane connection                                 |
| [VespaCloud](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespacloud) | ✅            | ❌         | `VespaCloud` is a control-plane connection to Vespa Cloud                                  |
| [VespaDocker](https://vespa-engine.github.io/pyvespa/api/vespa/deployment.md#vespa.deployment.VespaDocker) | ✅            | ❌         | `VespaDocker` is a control-plane connection to a Vespa server running in Docker            |
| [Vespa](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa)           | ❌            | ✅         | `Vespa` is a data-plane connection to an existing Vespa application                        |

## Defining your application[¶](#defining-your-application)

To initialize a connection to Vespa Cloud, you need to define your tenant name and application name.

In \[2\]:

Copied!

```
# Replace with your tenant name from the Vespa Cloud Console
tenant_name = "vespa-team"
# Replace with your application name (does not need to exist yet)
application = "authnotebook"
```

# Replace with your tenant name from the Vespa Cloud Console

tenant_name = "vespa-team"

# Replace with your application name (does not need to exist yet)

application = "authnotebook"

## Defining your application package[¶](#defining-your-application-package)

An [application package](https://docs.vespa.ai/en/application-packages.html) is the whole Vespa application configuration. It can either be constructed directly from python (as we will do below) or initalized from a path, for example by cloning a sample application from the [Vespa sample apps](https://github.com/vespa-engine/sample-apps).

Tip: You can use the command [vespa clone album-recommendation my-app](https://docs.vespa.ai/en/clients/vespa-cli.html) to clone a single sample app if you have the Vespa CLI installed.

For this guide, we will create a minimal application package. See other guides for more complex examples.

In \[3\]:

Copied!

```
from vespa.package import ApplicationPackage, Field, Schema, Document

schema_name = "doc"

schema = Schema(
    name=schema_name,
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary"]),
            Field(
                name="title",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="body",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
        ]
    ),
)

package = ApplicationPackage(name=application, schema=[schema])
```

from vespa.package import ApplicationPackage, Field, Schema, Document schema_name = "doc" schema = Schema( name=schema_name, document=Document( fields=\[ Field(name="id", type="string", indexing=["summary"]), Field( name="title", type="string", indexing=["index", "summary"], index="enable-bm25", ), Field( name="body", type="string", indexing=["index", "summary"], index="enable-bm25", ), \] ), ) package = ApplicationPackage(name=application, schema=[schema])

## Control-plane authentication[¶](#control-plane-authentication)

Next, we need to authenticate to the Vespa Cloud control-plane. There are two ways to authenticate to the control-plane:

### 1. **Interactive login**:[¶](#1-interactive-login)

This is the recommended way to authenticate to the control-plane. It opens a browser window for you to authenticate with either google or github.

This method does not work on windows, currently. You can run `vespa auth login` in a terminal to authenticate first, and then use this method (which will then reuse the generated token).

(We will not run this method here, as the notebook is run in CI, but you should run it in your local environment)

```
from vespa.deployment import VespaCloud

vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=application,
    application_package=package, # Could also initialize from application_root (path to application package)
)
```

You should see something similar to this:

```
log
Checking for access token in auth.json...
Access token expired. Please re-authenticate.
Your Device Confirmation code is: DRDT-ZZDC
Automatically open confirmation page in your default browser? [Y/n] y
Opened link in your browser: https://vespa.auth0.com/activate?user_code=DRDT-ZZDC
Waiting for login to complete in browser ... done;1m⣯
Success: Logged in
 auth.json created at /Users/thomas/.vespa/auth.json
Successfully obtained access token for control plane access.
```

### 2. **API-key authentication**[¶](#2-api-key-authentication)

This is a headless way to authenticate to the control-plane.

Note that the key must be generated, either with `vespa auth api-key` or in the Vespa Cloud console directly.

In \[4\]:

Copied!

```
from vespa.deployment import VespaCloud
from vespa.application import Vespa
import os

# Key is only used for CI/CD. Can be removed if logging in interactively
key = os.getenv("VESPA_TEAM_API_KEY", None)
if key is not None:
    key = key.replace(r"\n", "\n")  # To parse key correctly


vespa_cloud = VespaCloud(
    tenant=tenant_name,  # Note that the name cannot contain the characters `-` or `_`.
    application=application,
    key_content=key,  # Prefer to use  key_location="<path-to-key-file.pem>"
    application_package=package,
)
```

from vespa.deployment import VespaCloud from vespa.application import Vespa import os

# Key is only used for CI/CD. Can be removed if logging in interactively

key = os.getenv("VESPA_TEAM_API_KEY", None) if key is not None: key = key.replace(r"\\n", "\\n") # To parse key correctly vespa_cloud = VespaCloud( tenant=tenant_name, # Note that the name cannot contain the characters `-` or `_`. application=application, key_content=key, # Prefer to use key_location="\<path-to-key-file.pem>" application_package=package, )

```
Setting application...
Running: vespa config set application vespa-team.authnotebook
Setting target cloud...
Running: vespa config set target cloud

Api-key found for control plane access. Using api-key.
```

When you have authenticated to the control-plane of Vespa Cloud, key/cert for data-plane authentication will be generated automatically for you, if none exists.

The `data-plane-public-cert.pem` will be added to the application package (in `/security/clients.pem` directory) that will be deployed. You should keep them safe, as any app or users that need data-plane access to your Vespa application will need them.

For `dev`-deployments, we allow redeploying an application with a different key/cert than the previous deployment. For `prod`-deployments however, this is not allowed, and will require a `validation-overrides`-specification in the application package.

## Deploy to Vespa Cloud[¶](#deploy-to-vespa-cloud)

The app is now defined and ready to deploy to Vespa Cloud.

Deploy `package` to Vespa Cloud, by creating an instance of [VespaCloud](https://vespa-engine.github.io/pyvespa/api/vespa/deployment#VespaCloud):

The following will upload the application package to Vespa Cloud Dev Zone (`aws-us-east-1c`), read more about [Vespa Zones](https://cloud.vespa.ai/en/reference/zones.html). The Vespa Cloud Dev Zone is considered as a sandbox environment where resources are down-scaled and idle deployments are expired automatically. For information about production deployments, see the following [docs](https://cloud.vespa.ai/en/reference/deployment).

> Note: Deployments to dev and perf expire after 14 days of inactivity, i.e., 14 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 14 more days.

In \[5\]:

Copied!

```
app: Vespa = vespa_cloud.deploy()
```

app: Vespa = vespa_cloud.deploy()

```
Deployment started in run 1 of dev-aws-us-east-1c for vespa-team.authnotebook. This may take a few minutes the first time.
INFO    [06:35:26]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:35:27]  Using CA signed certificate version 1
INFO    [06:35:27]  Using 1 nodes in container cluster 'authnotebook_container'
INFO    [06:35:30]  Session 309490 for tenant 'vespa-team' prepared, but activation failed: 1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:35:33]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:35:33]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:35:42]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:35:42]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:35:52]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:35:52]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:36:03]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:36:03]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:36:14]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:36:14]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:36:22]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:36:22]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:36:33]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:36:33]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:36:42]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:36:42]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:36:52]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:36:53]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:37:03]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:37:03]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:37:12]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:37:12]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:37:22]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:37:22]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:37:33]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:37:33]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:37:43]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:37:43]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:37:53]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:37:54]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:38:03]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:38:03]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:38:12]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:38:12]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:38:22]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:38:22]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:38:33]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:38:34]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:38:42]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:38:43]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:38:52]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:38:53]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:39:02]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:39:03]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:39:12]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:39:13]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:39:22]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:39:22]  1/2 application hosts and 2/2 admin hosts for vespa-team.authnotebook have completed provisioning and bootstrapping, still waiting for h98840.dev.us-east-1c.aws.vespa-cloud.net
INFO    [06:39:34]  Deploying platform version 8.408.12 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [06:39:35]  Session 309490 for vespa-team.authnotebook.default activated
INFO    [06:39:56]  ######## Details for all nodes ########
INFO    [06:39:56]  h98612b.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:39:56]  --- platform vespa/cloud-tenant-rhel8:8.408.12
INFO    [06:39:56]  --- storagenode on port 19102 has not started 
INFO    [06:39:56]  --- searchnode on port 19107 has not started 
INFO    [06:39:56]  --- distributor on port 19111 has not started 
INFO    [06:39:56]  --- metricsproxy-container on port 19092 has not started 
INFO    [06:39:56]  h97566a.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:39:56]  --- platform vespa/cloud-tenant-rhel8:8.408.12
INFO    [06:39:56]  --- logserver-container on port 4080 has not started 
INFO    [06:39:56]  --- metricsproxy-container on port 19092 has not started 
INFO    [06:39:56]  h98840a.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:39:56]  --- platform vespa/cloud-tenant-rhel8:8.408.12
INFO    [06:39:56]  --- container on port 4080 has not started 
INFO    [06:39:56]  --- metricsproxy-container on port 19092 has not started 
INFO    [06:39:56]  h98621d.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:39:56]  --- platform vespa/cloud-tenant-rhel8:8.408.12
INFO    [06:39:56]  --- container-clustercontroller on port 19050 has not started 
INFO    [06:39:56]  --- metricsproxy-container on port 19092 has not started 
INFO    [06:40:33]  Found endpoints:
INFO    [06:40:33]  - dev.aws-us-east-1c
INFO    [06:40:33]   |-- https://ea8555a9.c6970ada.z.vespa-app.cloud/ (cluster 'authnotebook_container')
INFO    [06:40:33]  Deployment complete!
Only region: aws-us-east-1c available in dev environment.
Found mtls endpoint for authnotebook_container
URL: https://ea8555a9.c6970ada.z.vespa-app.cloud/
Application is up!
```

If the deployment failed, it is possible you forgot to add the key in the Vespa Cloud Console in the `vespa auth api-key` step above.

If you can authenticate, you should see lines like the following

```
 Deployment started in run 1 of dev-aws-us-east-1c for mytenant.authdemo.
```

The deployment takes a few minutes the first time while Vespa Cloud sets up the resources for your Vespa application

`app` now holds a reference to a [Vespa](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa) instance. We can access the mTLS protected endpoint name using the control-plane (vespa_cloud) instance. This endpoint we can query and feed to (data plane access) using the mTLS certificate generated in previous steps.

In \[6\]:

Copied!

```
mtls_endpoint = vespa_cloud.get_mtls_endpoint()
mtls_endpoint
```

mtls_endpoint = vespa_cloud.get_mtls_endpoint() mtls_endpoint

```
Found mtls endpoint for authnotebook_container
URL: https://ea8555a9.c6970ada.z.vespa-app.cloud/
```

Out\[6\]:

```
'https://ea8555a9.c6970ada.z.vespa-app.cloud/'
```

## Data-plane authentication[¶](#data-plane-authentication)

As we have mentioned, there are two ways to authenticate to the data-plane:

### 1. **mTLS - Certificate authentication**[¶](#1-mtls-certificate-authentication)

This is the default way to authenticate to the data-plane. It uses the certificate which was added to the application package upon deployment.

### 2. **Token-based authentication**[¶](#2-token-based-authentication)

A more convenient way to authenticate to the data-plane is to use a token. A token must be generated in the Vespa Cloud console. For more details, see the [Security Guide](https://cloud.vespa.ai/en/security/guide#configure-tokens)

Set a reasonable expiry, and copy the token to a safe place, such as for instance a passwordmanager. You will not be able to see it again.

After the token is generated, you need to add it as an auth-client to the application you want to access.

In pyvespa, this is done by adding the AuthClients to the application package:

**NB! - The method below applies to `dev`**

The approach described above applies to `dev`-deployments. For `prod`-deployments, it is a little more complex, and you need to add the `AuthClients` to your application package like this:

```
from vespa.package import ContainerCluster

auth_clients = [
            AuthClient(
                id="mtls",
                permissions=["read"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="token",
                permissions=["read"], # Set the permissions you need
                parameters=[Parameter("token", {"id": CLIENT_TOKEN_ID})],
            ),
        ]
# Add prod deployment config
prod_region = "aws-us-east-1c"
clusters = [
    ContentCluster(
        id=f"{schema_name}_content",
        nodes=Nodes(count="2"),
        document_name=schema_name,
        min_redundancy="2",
    ),
    ContainerCluster(
        id=f"{schema_name}_container",
        nodes=Nodes(count="2"),
        auth_clients=auth_clients, # Note that the auth_clients are added here for prod deployments
    ),
]
prod_region = "aws-us-east-1c"
deployment_config = DeploymentConfiguration(
    environment="prod", regions=[prod_region]
)
app_package = ApplicationPackage(name=application, schema=[schema], clusters=clusters, deployment=deployment_config)
```

See [Application Package reference](https://cloud.vespa.ai/en/reference/application-package) for more details.

In \[7\]:

Copied!

```
from vespa.package import AuthClient, Parameter

CLIENT_TOKEN_ID = "pyvespa_integration"
# Same as token name from the Vespa Cloud Console
auth_clients = [
    AuthClient(
        id="mtls",  # Note that you still need to include the mtls client.
        permissions=["read", "write"],
        parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
    ),
    AuthClient(
        id="token",
        permissions=["read"],
        parameters=[Parameter("token", {"id": CLIENT_TOKEN_ID})],
    ),
]

app_package = ApplicationPackage(
    name=application, schema=[schema], auth_clients=auth_clients
)
```

from vespa.package import AuthClient, Parameter CLIENT_TOKEN_ID = "pyvespa_integration"

# Same as token name from the Vespa Cloud Console

auth_clients = \[ AuthClient( id="mtls", # Note that you still need to include the mtls client. permissions=["read", "write"], parameters=[Parameter("certificate", {"file": "security/clients.pem"})], ), AuthClient( id="token", permissions=["read"], parameters=[Parameter("token", {"id": CLIENT_TOKEN_ID})], ), \] app_package = ApplicationPackage( name=application, schema=[schema], auth_clients=auth_clients )

Notice that we added the `read` and `write` permissions to mtls clients, and only `read` to the token client.

Make sure to restrict the permissions to suit your needs.

Now, we can deploy a new instance of the application package with the new auth-client added:

See [Tenants, apps, instances](https://cloud.vespa.ai/en/tenant-apps-instances) for details on terminology for Vespa Cloud.

In \[8\]:

Copied!

```
instance = "token"

vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=application,
    key_content=key,
    application_package=app_package,
)
app = vespa_cloud.deploy(instance=instance)
```

instance = "token" vespa_cloud = VespaCloud( tenant=tenant_name, application=application, key_content=key, application_package=app_package, ) app = vespa_cloud.deploy(instance=instance)

```
Setting application...
Running: vespa config set application vespa-team.authnotebook
Setting target cloud...
Running: vespa config set target cloud

Api-key found for control plane access. Using api-key.
Deployment started in run 60 of dev-aws-us-east-1c for vespa-team.authnotebook.token. This may take a few minutes the first time.
INFO    [06:40:38]  Deploying platform version 8.408.12 and application dev build 54 for dev-aws-us-east-1c of token ...
INFO    [06:40:39]  Using CA signed certificate version 1
INFO    [06:40:39]  Using 1 nodes in container cluster 'authnotebook_container'
WARNING [06:40:41]  Auto-overriding validation which would be disallowed in production: certificate-removal: Data plane certificate(s) from cluster 'authnotebook_container' is removed (removed certificates: [CN=cloud.vespa.example]) This can cause client connection issues.. To allow this add <allow until='yyyy-mm-dd'>certificate-removal</allow> to validation-overrides.xml, see https://docs.vespa.ai/en/reference/validation-overrides.html
INFO    [06:40:42]  Session 309492 for tenant 'vespa-team' prepared and activated.
INFO    [06:40:43]  ######## Details for all nodes ########
INFO    [06:40:43]  h97526a.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:40:43]  --- platform vespa/cloud-tenant-rhel8:8.408.12
INFO    [06:40:43]  --- storagenode on port 19102 has config generation 309488, wanted is 309492
INFO    [06:40:43]  --- searchnode on port 19107 has config generation 309488, wanted is 309492
INFO    [06:40:43]  --- distributor on port 19111 has config generation 309488, wanted is 309492
INFO    [06:40:43]  --- metricsproxy-container on port 19092 has config generation 309488, wanted is 309492
INFO    [06:40:43]  h97566b.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:40:43]  --- platform vespa/cloud-tenant-rhel8:8.408.12
INFO    [06:40:43]  --- logserver-container on port 4080 has config generation 309488, wanted is 309492
INFO    [06:40:43]  --- metricsproxy-container on port 19092 has config generation 309488, wanted is 309492
INFO    [06:40:43]  h97538e.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:40:43]  --- platform vespa/cloud-tenant-rhel8:8.408.12
INFO    [06:40:43]  --- container-clustercontroller on port 19050 has config generation 309492, wanted is 309492
INFO    [06:40:43]  --- metricsproxy-container on port 19092 has config generation 309488, wanted is 309492
INFO    [06:40:43]  h97567a.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:40:43]  --- platform vespa/cloud-tenant-rhel8:8.408.12
INFO    [06:40:43]  --- container on port 4080 has config generation 309488, wanted is 309492
INFO    [06:40:43]  --- metricsproxy-container on port 19092 has config generation 309488, wanted is 309492
INFO    [06:40:53]  Found endpoints:
INFO    [06:40:53]  - dev.aws-us-east-1c
INFO    [06:40:53]   |-- https://ab50e0c2.c6970ada.z.vespa-app.cloud/ (cluster 'authnotebook_container')
INFO    [06:40:53]  Deployment of new application complete!
Only region: aws-us-east-1c available in dev environment.
Found mtls endpoint for authnotebook_container
URL: https://ab50e0c2.c6970ada.z.vespa-app.cloud/
Application is up!
```

Note that the connection that will be returned by default, will be the mTLS connection. If you want to get a connection using token-based authentication, you can do it like this:

In \[9\]:

Copied!

```
token_app = vespa_cloud.get_application(
    instance=instance,
    endpoint_type="token",
    vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN"),
)
```

token_app = vespa_cloud.get_application( instance=instance, endpoint_type="token", vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN"), )

```
Only region: aws-us-east-1c available in dev environment.
Found token endpoint for authnotebook_container
URL: https://c7f94a93.c6970ada.z.vespa-app.cloud/
Application is up!
```

In \[10\]:

Copied!

```
token_app.get_application_status()
```

token_app.get_application_status()

Out\[10\]:

```
<Response [200]>
```

Note that a Vespa application creates a separate URL endpoint for each auth-client added. Here is how you can retrieve the URL for the token endpoint:

In \[11\]:

Copied!

```
token_endpoint = vespa_cloud.get_token_endpoint(instance=instance)
token_endpoint
```

token_endpoint = vespa_cloud.get_token_endpoint(instance=instance) token_endpoint

```
Found token endpoint for authnotebook_container
URL: https://c7f94a93.c6970ada.z.vespa-app.cloud/
```

Out\[11\]:

```
'https://c7f94a93.c6970ada.z.vespa-app.cloud/'
```

## Re-connecting to a deployed application[¶](#re-connecting-to-a-deployed-application)

To connect to a deployed application, you can use the `Vespa` class, which is a data-plane connection to an existing Vespa application.

The `Vespa` class requires the endpoint URL.

Note that this class can also be instantiated without authentication, typically used if connecting to an instance running in Docker, see [VespaDocker](https://vespa-engine.github.io/pyvespa/api/vespa/deployment.md#vespa.deployment.VespaDocker).

### Connecting using mTLS[¶](#connecting-using-mtls)

To connect to the Vespa application using mTLS, you must pass `key` and `cert` to the `Vespa` class. Both should be a path to the respective files, matching the cert that was added to the application package upon deployment.

A common error is to try to regenerate the key/cert after deployment, causing a mismatch between the key/cert you are trying to authenticate with, and the cert added to the application package.

In \[12\]:

Copied!

```
import os

# Get user home directory
home = os.path.expanduser("~")
# Vespa key/cert directory
app_dir = f"{home}/.vespa/{tenant_name}.{application}.default/"

cert_path = f"{app_dir}/data-plane-public-cert.pem"
key_path = f"{app_dir}/data-plane-private-key.pem"
```

import os

# Get user home directory

home = os.path.expanduser("~")

# Vespa key/cert directory

app_dir = f"{home}/.vespa/{tenant_name}.{application}.default/" cert_path = f"{app_dir}/data-plane-public-cert.pem" key_path = f"{app_dir}/data-plane-private-key.pem"

In \[13\]:

Copied!

```
from vespa.application import Vespa

app = Vespa(url=mtls_endpoint, cert=cert_path, key=key_path)
app.get_application_status()
```

from vespa.application import Vespa app = Vespa(url=mtls_endpoint, cert=cert_path, key=key_path) app.get_application_status()

Out\[13\]:

```
<Response [200]>
```

#### Using `requests`[¶](#using-requests)

It is often overlooked that all interactions with Vespa are through HTTP-api calls, so you are free to use any HTTP client you like.

Below is an example of how to use the `requests` library to interact with Vespa, using `key` and `cert` for authentication, and the [/document/v1/](https://docs.vespa.ai/en/reference/document-v1-api-reference.html) endpoint to feed data to Vespa.

In \[14\]:

Copied!

```
import requests

session = requests.Session()
session.cert = (cert_path, key_path)
url = f"{mtls_endpoint}/document/v1/doc/doc/docid/1"
data = {
    "fields": {
        "id": "id:doc:doc::1",
        "title": "the title",
        "body": "the body",
    }
}
resp = session.post(url, json=data).json()
resp
```

import requests session = requests.Session() session.cert = (cert_path, key_path) url = f"{mtls_endpoint}/document/v1/doc/doc/docid/1" data = { "fields": { "id": "id:doc:doc::1", "title": "the title", "body": "the body", } } resp = session.post(url, json=data).json() resp

Out\[14\]:

```
{'pathId': '/document/v1/doc/doc/docid/1', 'id': 'id:doc:doc::1'}
```

## Connecting using token[¶](#connecting-using-token)

To connect to the Vespa application using a token, you must pass the token value to the `Vespa` class as `vespa_cloud_secret_token`.

In \[15\]:

Copied!

```
app = Vespa(
    url=token_endpoint, vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN")
)
app.get_application_status()
```

app = Vespa( url=token_endpoint, vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN") ) app.get_application_status()

Out\[15\]:

```
<Response [200]>
```

### Using cURL[¶](#using-curl)

Token authentication provides an even more convenient way to authenticate to the data-plane, as you do not need to handle key/cert files, and can just add the token to the HTTP header, as shown in the example below.

```
curl -H "Authorization: Bearer $TOKEN" https://{endpoint}/document/v1/{document-type}/{document-id}
```

## Next steps[¶](#next-steps)

This was a guide to the different authentication methods when interacting with Vespa Cloud for different purposes.

Try to deploy a frontend as interface to your Vespa application.

Example of some providers are:

- [Cloudflare Workers](https://workers.cloudflare.com/), see also <https://cloud.vespa.ai/en/security/cloudflare-workers.html>
- [Vercel](https://vercel.com/)
- [Railway](https://railway.app/) etc.

## Cleanup[¶](#cleanup)

In \[16\]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()

```
Deactivated vespa-team.authnotebook in dev.aws-us-east-1c
Deleted instance vespa-team.authnotebook.default
```
