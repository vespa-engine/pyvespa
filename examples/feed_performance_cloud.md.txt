# Feeding to Vespa Cloud[¶](#feeding-to-vespa-cloud)

Our [previous notebook](https://vespa-engine.github.io/pyvespa/examples/feed_performance.md), we demonstrated one way of benchmarking feed performance to a local Vespa instance running in Docker. In this notebook, we will look at the same methods but how feeding to [Vespa Cloud](https://cloud.vespa.ai) affects the performance of the different methods.

The key difference between feeding to a local Vespa instance and a Vespa Cloud instance is the network latency. Additionally, we will introduce embedding in Vespa at feed time, which is a realistic scenario for many use cases.

We will look at these 3 different methods:

1. Using `feed_iterable()` - which uses threading to parallelize the feed operation. Best for CPU-bound operations.
1. Using `feed_async_iterable()` - which uses asyncio to parallelize the feed operation. Also uses `httpx` with HTTP/2-support. Performs best for IO-bound operations.
1. Using [Vespa CLI](https://docs.vespa.ai/en/vespa-cli).

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html). The `vespacli` python package is just a thin wrapper, allowing for installation through pypi.

> Do NOT install if you already have the Vespa CLI installed.

[Install pyvespa](https://pyvespa.readthedocs.io/), and other dependencies.

In \[1\]:

Copied!

```
!pip3 install vespacli pyvespa datasets plotly>=5.20
```

!pip3 install vespacli pyvespa datasets plotly>=5.20

```
zsh:1: 5.20 not found
```

## Create an application package[¶](#create-an-application-package)

The [application package](https://vespa-engine.github.io/pyvespa/api/vespa/package.md) has all the Vespa configuration files.

For this demo, we will use a simple application package

In \[2\]:

Copied!

```
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    FieldSet,
    HNSW,
)

# Define the application name (can NOT contain `_` or `-`)

application = "feedperformancecloud"


package = ApplicationPackage(
    name=application,
    schema=[
        Schema(
            name="doc",
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["summary"]),
                    Field(name="text", type="string", indexing=["index", "summary"]),
                    Field(
                        name="embedding",
                        type="tensor<float>(x[1024])",
                        # Note that we are NOT embedding with a vespa model here, but that is also possible.
                        indexing=["summary", "attribute", "index"],
                        ann=HNSW(distance_metric="angular"),
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["text"])],
        )
    ],
)
```

from vespa.package import ( ApplicationPackage, Field, Schema, Document, FieldSet, HNSW, )

# Define the application name (can NOT contain `_` or `-`)

application = "feedperformancecloud" package = ApplicationPackage( name=application, schema=\[ Schema( name="doc", document=Document( fields=\[ Field(name="id", type="string", indexing=["summary"]), Field(name="text", type="string", indexing=["index", "summary"]), Field( name="embedding", type="tensor<float>(x[1024])",

# Note that we are NOT embedding with a vespa model here, but that is also possible.

indexing=["summary", "attribute", "index"], ann=HNSW(distance_metric="angular"), ), \] ), fieldsets=\[FieldSet(name="default", fields=["text"])\], ) \], )

Note that the `ApplicationPackage` name cannot have `-` or `_`.

## Deploy the Vespa application[¶](#deploy-the-vespa-application)

Deploy `package` on the local machine using Docker, without leaving the notebook, by creating an instance of [VespaDocker](https://vespa-engine.github.io/pyvespa/api/vespa/deployment#vespa.deployment.VespaDocker). `VespaDocker` connects to the local Docker daemon socket and starts the [Vespa docker image](https://hub.docker.com/r/vespaengine/vespa/).

If this step fails, please check that the Docker daemon is running, and that the Docker daemon socket can be used by clients (Configurable under advanced settings in Docker Desktop).

Follow the instructions from the output above and add the control-plane key in the console at `https://console.vespa-cloud.com/tenant/TENANT_NAME/account/keys` (replace TENANT_NAME with your tenant name).

In \[3\]:

Copied!

```
from vespa.deployment import VespaCloud
import os

# Replace with your tenant name from the Vespa Cloud Console
tenant_name = "vespa-team"

# Key is only used for CI/CD. Can be removed if logging in interactively
key = os.getenv("VESPA_TEAM_API_KEY", None)
if key is not None:
    key = key.replace(r"\n", "\n")  # To parse key correctly


vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=application,
    key_content=key,  # Key is only used for CI/CD. Can be removed if logging in interactively
    application_package=package,
)
```

from vespa.deployment import VespaCloud import os

# Replace with your tenant name from the Vespa Cloud Console

tenant_name = "vespa-team"

# Key is only used for CI/CD. Can be removed if logging in interactively

key = os.getenv("VESPA_TEAM_API_KEY", None) if key is not None: key = key.replace(r"\\n", "\\n") # To parse key correctly vespa_cloud = VespaCloud( tenant=tenant_name, application=application, key_content=key, # Key is only used for CI/CD. Can be removed if logging in interactively application_package=package, )

```
Setting application...
Running: vespa config set application vespa-team.feedperformancecloud
Setting target cloud...
Running: vespa config set target cloud

Api-key found for control plane access. Using api-key.
```

`app` now holds a reference to a [VespaCloud](https://vespa-engine.github.io/pyvespa/api/vespa/deployment#VespaCloud) instance.

In \[4\]:

Copied!

```
from vespa.application import Vespa

app: Vespa = vespa_cloud.deploy()
```

from vespa.application import Vespa app: Vespa = vespa_cloud.deploy()

```
Deployment started in run 9 of dev-aws-us-east-1c for vespa-team.feedperformancecloud. This may take a few minutes the first time.
INFO    [07:22:29]  Deploying platform version 8.392.14 and application dev build 7 for dev-aws-us-east-1c of default ...
INFO    [07:22:30]  Using CA signed certificate version 1
INFO    [07:22:30]  Using 1 nodes in container cluster 'feedperformancecloud_container'
WARNING [07:22:33]  Auto-overriding validation which would be disallowed in production: certificate-removal: Data plane certificate(s) from cluster 'feedperformancecloud_container' is removed (removed certificates: [CN=cloud.vespa.example]) This can cause client connection issues.. To allow this add <allow until='yyyy-mm-dd'>certificate-removal</allow> to validation-overrides.xml, see https://docs.vespa.ai/en/reference/validation-overrides.html
INFO    [07:22:34]  Session 304192 for tenant 'vespa-team' prepared and activated.
INFO    [07:22:35]  ######## Details for all nodes ########
INFO    [07:22:35]  h95731a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [07:22:35]  --- platform vespa/cloud-tenant-rhel8:8.392.14
INFO    [07:22:35]  --- container on port 4080 has not started 
INFO    [07:22:35]  --- metricsproxy-container on port 19092 has config generation 304192, wanted is 304192
INFO    [07:22:35]  h95729b.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [07:22:35]  --- platform vespa/cloud-tenant-rhel8:8.392.14
INFO    [07:22:35]  --- storagenode on port 19102 has config generation 304192, wanted is 304192
INFO    [07:22:35]  --- searchnode on port 19107 has config generation 304192, wanted is 304192
INFO    [07:22:35]  --- distributor on port 19111 has config generation 304192, wanted is 304192
INFO    [07:22:35]  --- metricsproxy-container on port 19092 has config generation 304192, wanted is 304192
INFO    [07:22:35]  h93272g.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [07:22:35]  --- platform vespa/cloud-tenant-rhel8:8.392.14
INFO    [07:22:35]  --- logserver-container on port 4080 has config generation 304192, wanted is 304192
INFO    [07:22:35]  --- metricsproxy-container on port 19092 has config generation 304192, wanted is 304192
INFO    [07:22:35]  h93272h.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [07:22:35]  --- platform vespa/cloud-tenant-rhel8:8.392.14
INFO    [07:22:35]  --- container-clustercontroller on port 19050 has config generation 304192, wanted is 304192
INFO    [07:22:35]  --- metricsproxy-container on port 19092 has config generation 304192, wanted is 304192
INFO    [07:22:42]  Found endpoints:
INFO    [07:22:42]  - dev.aws-us-east-1c
INFO    [07:22:42]   |-- https://b48e8812.bc737822.z.vespa-app.cloud/ (cluster 'feedperformancecloud_container')
INFO    [07:22:44]  Deployment of new application complete!
Found mtls endpoint for feedperformancecloud_container
URL: https://b48e8812.bc737822.z.vespa-app.cloud/
Connecting to https://b48e8812.bc737822.z.vespa-app.cloud/
Using mtls_key_cert Authentication against endpoint https://b48e8812.bc737822.z.vespa-app.cloud//ApplicationStatus
Application is up!
Finished deployment.
```

Note that if you already have a Vespa Cloud instance running, the recommended way to initialize a `Vespa` instance is directly, by passing the `endpoint` and `tenant` parameters to the `Vespa` constructor, along with either:

1. Key/cert for dataplane authentication (generated as part of deployment, copied into the application package, in `/security/clients.pem`, and `~/.vespa/mytenant.myapplication/data-plane-public-cert.pem` and `~/.vespa/mytenant.myapplication/data-plane-private-key.pem`).

```
from vespa.application import Vespa

app: Vespa = Vespa(
    url="https://my-endpoint.z.vespa-app.cloud",
    tenant="my-tenant",
    key_file="path/to/private-key.pem",
    cert_file="path/to/certificate.pem",
)
```

2. Using a token (must be generated in [Vespa Cloud Console](https://console.vespa-cloud.com/) and defined in the application package, see <https://cloud.vespa.ai/en/security/guide>.

```
from vespa.application import Vespa
import os

app: Vespa = Vespa(
    url="https://my-endpoint.z.vespa-app.cloud",
    tenant="my-tenant",
    vespa_cloud_secret_token=os.getenv("VESPA_CLOUD_SECRET_TOKEN"),
)
```

In \[5\]:

Copied!

```
app.get_application_status()
```

app.get_application_status()

```
Using mtls_key_cert Authentication against endpoint https://b48e8812.bc737822.z.vespa-app.cloud//ApplicationStatus
```

Out\[5\]:

```
<Response [200]>
```

## Preparing the data[¶](#preparing-the-data)

In this example we use [HF Datasets](https://huggingface.co/docs/datasets/index) library to stream the ["Cohere/wikipedia-2023-11-embed-multilingual-v3"](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3) dataset and index in our newly deployed Vespa instance.

The dataset contains Wikipedia-pages, and their corresponding embeddings.

> For this exploration, we will use the `id`, `text` and `embedding`-fields

The following uses the [stream](https://huggingface.co/docs/datasets/stream) option of datasets to stream the data without downloading all the contents locally.

The `map` functionality allows us to convert the dataset fields into the expected feed format for `pyvespa` which expects a dict with the keys `id` and `fields`:

`{ "id": "vespa-document-id", "fields": {"vespa_field": "vespa-field-value"}}`

In \[ \]:

Copied!

```
from datasets import load_dataset
```

from datasets import load_dataset

## Utility function to create a dataset with different number of documents[¶](#utility-function-to-create-a-dataset-with-different-number-of-documents)

In \[7\]:

Copied!

```
def get_dataset(n_docs: int = 1000):
    dataset = load_dataset(
        "Cohere/wikipedia-2023-11-embed-multilingual-v3",
        "simple",
        split=f"train[:{n_docs}]",
    )
    dataset = dataset.map(
        lambda x: {
            "id": x["_id"] + "-iter",
            "fields": {"text": x["text"], "embedding": x["emb"]},
        }
    ).select_columns(["id", "fields"])
    return dataset
```

def get_dataset(n_docs: int = 1000): dataset = load_dataset( "Cohere/wikipedia-2023-11-embed-multilingual-v3", "simple", split=f"train[:{n_docs}]", ) dataset = dataset.map( lambda x: { "id": x["\_id"] + "-iter", "fields": {"text": x["text"], "embedding": x["emb"]}, } ).select_columns(["id", "fields"]) return dataset

### A dataclass to store the parameters and results of the different feeding methods[¶](#a-dataclass-to-store-the-parameters-and-results-of-the-different-feeding-methods)

In \[8\]:

Copied!

```
from dataclasses import dataclass
from typing import Callable, Optional, Iterable, Dict


@dataclass
class FeedParams:
    name: str
    num_docs: int
    max_connections: int
    function_name: str
    max_workers: Optional[int] = None
    max_queue_size: Optional[int] = None


@dataclass
class FeedResult(FeedParams):
    feed_time: Optional[float] = None
```

from dataclasses import dataclass from typing import Callable, Optional, Iterable, Dict @dataclass class FeedParams: name: str num_docs: int max_connections: int function_name: str max_workers: Optional[int] = None max_queue_size: Optional[int] = None @dataclass class FeedResult(FeedParams): feed_time: Optional[float] = None

### A common callback function to notify if something goes wrong[¶](#a-common-callback-function-to-notify-if-something-goes-wrong)

In \[9\]:

Copied!

```
from vespa.io import VespaResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
        )
```

from vespa.io import VespaResponse def callback(response: VespaResponse, id: str): if not response.is_successful(): print( f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}" )

### Defining our feeding functions[¶](#defining-our-feeding-functions)

In \[10\]:

Copied!

```
import time
import asyncio
from vespa.application import Vespa
```

import time import asyncio from vespa.application import Vespa

In \[11\]:

Copied!

```
def feed_iterable(app: Vespa, params: FeedParams, data: Iterable[Dict]) -> FeedResult:
    start = time.time()
    app.feed_iterable(
        data,
        schema="doc",
        namespace="pyvespa-feed",
        operation_type="feed",
        max_queue_size=params.max_queue_size,
        max_workers=params.max_workers,
        max_connections=params.max_connections,
        callback=callback,
    )
    end = time.time()
    sync_feed_time = end - start
    return FeedResult(
        **params.__dict__,
        feed_time=sync_feed_time,
    )


def feed_async_iterable(
    app: Vespa, params: FeedParams, data: Iterable[Dict]
) -> FeedResult:
    start = time.time()
    app.feed_async_iterable(
        data,
        schema="doc",
        namespace="pyvespa-feed",
        operation_type="feed",
        max_queue_size=params.max_queue_size,
        max_workers=params.max_workers,
        max_connections=params.max_connections,
        callback=callback,
    )
    end = time.time()
    sync_feed_time = end - start
    return FeedResult(
        **params.__dict__,
        feed_time=sync_feed_time,
    )
```

def feed_iterable(app: Vespa, params: FeedParams, data: Iterable[Dict]) -> FeedResult: start = time.time() app.feed_iterable( data, schema="doc", namespace="pyvespa-feed", operation_type="feed", max_queue_size=params.max_queue_size, max_workers=params.max_workers, max_connections=params.max_connections, callback=callback, ) end = time.time() sync_feed_time = end - start return FeedResult( \*\*params.__dict__, feed_time=sync_feed_time, ) def feed_async_iterable( app: Vespa, params: FeedParams, data: Iterable[Dict] ) -> FeedResult: start = time.time() app.feed_async_iterable( data, schema="doc", namespace="pyvespa-feed", operation_type="feed", max_queue_size=params.max_queue_size, max_workers=params.max_workers, max_connections=params.max_connections, callback=callback, ) end = time.time() sync_feed_time = end - start return FeedResult( \*\*params.__dict__, feed_time=sync_feed_time, )

## Defining our hyperparameters[¶](#defining-our-hyperparameters)

In \[12\]:

Copied!

```
from itertools import product

# We will only run for up to 10 000 documents here as notebook is run as part of CI.

num_docs = [
    1000,
    5_000,
    10_000,
]
params_by_function = {
    "feed_async_iterable": {
        "num_docs": num_docs,
        "max_connections": [1],
        "max_workers": [64],
        "max_queue_size": [2500],
    },
    "feed_iterable": {
        "num_docs": num_docs,
        "max_connections": [64],
        "max_workers": [64],
        "max_queue_size": [2500],
    },
}

feed_params = []
# Create one FeedParams instance of each permutation
for func, parameters in params_by_function.items():
    print(f"Function: {func}")
    keys, values = zip(*parameters.items())
    for combination in product(*values):
        settings = dict(zip(keys, combination))
        print(settings)
        feed_params.append(
            FeedParams(
                name=f"{settings['num_docs']}_{settings['max_connections']}_{settings.get('max_workers', 0)}_{func}",
                function_name=func,
                **settings,
            )
        )
    print("\n")  # Just to add space between different functions
```

from itertools import product

# We will only run for up to 10 000 documents here as notebook is run as part of CI.

num_docs = [ 1000, 5_000, 10_000, ] params_by_function = { "feed_async_iterable": { "num_docs": num_docs, "max_connections": [1], "max_workers": [64], "max_queue_size": [2500], }, "feed_iterable": { "num_docs": num_docs, "max_connections": [64], "max_workers": [64], "max_queue_size": [2500], }, } feed_params = []

# Create one FeedParams instance of each permutation

for func, parameters in params_by_function.items(): print(f"Function: {func}") keys, values = zip(\*parameters.items()) for combination in product(\*values): settings = dict(zip(keys, combination)) print(settings) feed_params.append( FeedParams( name=f"{settings['num_docs']}_{settings['max_connections']}_{settings.get('max_workers', 0)}\_{func}", function_name=func, \*\*settings, ) ) print("\\n") # Just to add space between different functions

```
Function: feed_async_iterable
{'num_docs': 1000, 'max_connections': 1, 'max_workers': 64, 'max_queue_size': 2500}
{'num_docs': 5000, 'max_connections': 1, 'max_workers': 64, 'max_queue_size': 2500}
{'num_docs': 10000, 'max_connections': 1, 'max_workers': 64, 'max_queue_size': 2500}


Function: feed_iterable
{'num_docs': 1000, 'max_connections': 64, 'max_workers': 64, 'max_queue_size': 2500}
{'num_docs': 5000, 'max_connections': 64, 'max_workers': 64, 'max_queue_size': 2500}
{'num_docs': 10000, 'max_connections': 64, 'max_workers': 64, 'max_queue_size': 2500}
```

In \[13\]:

Copied!

```
print(f"Total number of feed_params: {len(feed_params)}")
```

print(f"Total number of feed_params: {len(feed_params)}")

```
Total number of feed_params: 6
```

Now, we will need a way to retrieve the callable function from the function name.

In \[14\]:

Copied!

```
# Get reference to function from string name
def get_func_from_str(func_name: str) -> Callable:
    return globals()[func_name]
```

# Get reference to function from string name

def get_func_from_str(func_name: str) -> Callable: return globals()[func_name]

### Function to clean up after each feed[¶](#function-to-clean-up-after-each-feed)

For a fair comparison, we will delete the data before feeding it again.

In \[15\]:

Copied!

```
from typing import Iterable, Dict
from vespa.application import Vespa


def delete_data(app: Vespa, data: Iterable[Dict]):
    app.feed_iterable(
        iter=data,
        schema="doc",
        namespace="pyvespa-feed",
        operation_type="delete",
        callback=callback,
        max_workers=16,
        max_connections=16,
    )
```

from typing import Iterable, Dict from vespa.application import Vespa def delete_data(app: Vespa, data: Iterable[Dict]): app.feed_iterable( iter=data, schema="doc", namespace="pyvespa-feed", operation_type="delete", callback=callback, max_workers=16, max_connections=16, )

## Main experiment loop[¶](#main-experiment-loop)

The line below is used to make the code run in Jupyter, as it is already running an event loop

In \[16\]:

Copied!

```
import nest_asyncio

nest_asyncio.apply()
```

import nest_asyncio nest_asyncio.apply()

In \[17\]:

Copied!

```
results = []
for params in feed_params:
    print("-" * 50)
    print("Starting feed with params:")
    print(params)
    data = get_dataset(params.num_docs)
    if "xxx" not in params.function_name:
        if "feed_sync" in params.function_name:
            print("Skipping feed_sync")
            continue
        feed_result = get_func_from_str(params.function_name)(
            app=app, params=params, data=data
        )
    else:
        feed_result = asyncio.run(
            get_func_from_str(params.function_name)(app=app, params=params, data=data)
        )
    print(feed_result.feed_time)
    results.append(feed_result)
    print("Deleting data")
    time.sleep(3)
    delete_data(app, data)
```

results = [] for params in feed_params: print("-" * 50) print("Starting feed with params:") print(params) data = get_dataset(params.num_docs) if "xxx" not in params.function_name: if "feed_sync" in params.function_name: print("Skipping feed_sync") continue feed_result = get_func_from_str(params.function_name)( app=app, params=params, data=data ) else: feed_result = asyncio.run( get_func_from_str(params.function_name)(app=app, params=params, data=data) ) print(feed_result.feed_time) results.append(feed_result) print("Deleting data") time.sleep(3) delete_data(app, data)

```
--------------------------------------------------
Starting feed with params:
FeedParams(name='1000_1_64_feed_async_iterable', num_docs=1000, max_connections=1, function_name='feed_async_iterable', max_workers=64, max_queue_size=2500)
```

```
Using mtls_key_cert Authentication against endpoint https://b48e8812.bc737822.z.vespa-app.cloud//ApplicationStatus
7.062151908874512
Deleting data
--------------------------------------------------
Starting feed with params:
FeedParams(name='5000_1_64_feed_async_iterable', num_docs=5000, max_connections=1, function_name='feed_async_iterable', max_workers=64, max_queue_size=2500)
20.979923963546753
Deleting data
--------------------------------------------------
Starting feed with params:
FeedParams(name='10000_1_64_feed_async_iterable', num_docs=10000, max_connections=1, function_name='feed_async_iterable', max_workers=64, max_queue_size=2500)
41.321199893951416
Deleting data
--------------------------------------------------
Starting feed with params:
FeedParams(name='1000_64_64_feed_iterable', num_docs=1000, max_connections=64, function_name='feed_iterable', max_workers=64, max_queue_size=2500)
16.278107166290283
Deleting data
--------------------------------------------------
Starting feed with params:
FeedParams(name='5000_64_64_feed_iterable', num_docs=5000, max_connections=64, function_name='feed_iterable', max_workers=64, max_queue_size=2500)
78.27990508079529
Deleting data
--------------------------------------------------
Starting feed with params:
FeedParams(name='10000_64_64_feed_iterable', num_docs=10000, max_connections=64, function_name='feed_iterable', max_workers=64, max_queue_size=2500)
156.38266611099243
Deleting data
```

In \[18\]:

Copied!

```
# Create a pandas DataFrame with the results
import pandas as pd

df = pd.DataFrame([result.__dict__ for result in results])
df["requests_per_second"] = df["num_docs"] / df["feed_time"]
df
```

# Create a pandas DataFrame with the results

import pandas as pd df = pd.DataFrame(\[result.__dict__ for result in results\]) df["requests_per_second"] = df["num_docs"] / df["feed_time"] df

Out\[18\]:

|     | name                           | num_docs | max_connections | function_name       | max_workers | max_queue_size | feed_time  | requests_per_second |
| --- | ------------------------------ | -------- | --------------- | ------------------- | ----------- | -------------- | ---------- | ------------------- |
| 0   | 1000_1_64_feed_async_iterable  | 1000     | 1               | feed_async_iterable | 64          | 2500           | 7.062152   | 141.599899          |
| 1   | 5000_1_64_feed_async_iterable  | 5000     | 1               | feed_async_iterable | 64          | 2500           | 20.979924  | 238.323075          |
| 2   | 10000_1_64_feed_async_iterable | 10000    | 1               | feed_async_iterable | 64          | 2500           | 41.321200  | 242.006525          |
| 3   | 1000_64_64_feed_iterable       | 1000     | 64              | feed_iterable       | 64          | 2500           | 16.278107  | 61.432204           |
| 4   | 5000_64_64_feed_iterable       | 5000     | 64              | feed_iterable       | 64          | 2500           | 78.279905  | 63.873353           |
| 5   | 10000_64_64_feed_iterable      | 10000    | 64              | feed_iterable       | 64          | 2500           | 156.382666 | 63.945706           |

## Plotting the results[¶](#plotting-the-results)

Let's plot the results to see how the different methods compare.

In \[19\]:

Copied!

```
import plotly.express as px


def plot_performance(df: pd.DataFrame):
    # Create a scatter plot with logarithmic scale for both axes using Plotly Express
    fig = px.scatter(
        df,
        x="num_docs",
        y="requests_per_second",
        color="function_name",  # Defines color based on different functions
        log_x=True,  # Set x-axis to logarithmic scale
        log_y=False,  # If you also want the y-axis in logarithmic scale, set this to True
        title="Performance: Requests per Second vs. Number of Documents",
        labels={  # Customizing axis labels
            "num_docs": "Number of Documents",
            "requests_per_second": "Requests per Second",
            "max_workers": "max_workers",
            "max_queue_size": "max_queue_size",
        },
        template="plotly_white",  # This sets the style to a white background, adhering to Tufte's minimalist principles
        hover_data=[
            "max_workers",
            "max_queue_size",
            "max_connections",
        ],  # Additional information to show on hover
    )

    # Update layout for better readability, similar to 'talk' context in Seaborn
    fig.update_layout(
        font=dict(
            size=16,  # Adjusting font size for better visibility, similar to 'talk' context
        ),
        legend_title_text="Function Details",  # Custom legend title
        legend=dict(
            title_font_size=16,
            x=800,  # Adjusting legend position similar to bbox_to_anchor in Matplotlib
            xanchor="auto",
            y=1,
            yanchor="auto",
        ),
        width=800,  # Adjusting width of the plot
    )
    fig.update_xaxes(
        tickvals=[1000, 5000, 10000],  # Set specific tick values
        ticktext=["1k", "5k", "10k"],  # Set corresponding tick labels
    )

    fig.update_traces(
        marker=dict(size=12, opacity=0.7)
    )  # Adjust marker size and opacity
    # Show plot
    fig.show()
    # Save plot as HTML file
    fig.write_html("performance.html")


plot_performance(df)
```

import plotly.express as px def plot_performance(df: pd.DataFrame):

# Create a scatter plot with logarithmic scale for both axes using Plotly Express

fig = px.scatter( df, x="num_docs", y="requests_per_second", color="function_name", # Defines color based on different functions log_x=True, # Set x-axis to logarithmic scale log_y=False, # If you also want the y-axis in logarithmic scale, set this to True title="Performance: Requests per Second vs. Number of Documents", labels={ # Customizing axis labels "num_docs": "Number of Documents", "requests_per_second": "Requests per Second", "max_workers": "max_workers", "max_queue_size": "max_queue_size", }, template="plotly_white", # This sets the style to a white background, adhering to Tufte's minimalist principles hover_data=[ "max_workers", "max_queue_size", "max_connections", ], # Additional information to show on hover )

# Update layout for better readability, similar to 'talk' context in Seaborn

fig.update_layout( font=dict( size=16, # Adjusting font size for better visibility, similar to 'talk' context ), legend_title_text="Function Details", # Custom legend title legend=dict( title_font_size=16, x=800, # Adjusting legend position similar to bbox_to_anchor in Matplotlib xanchor="auto", y=1, yanchor="auto", ), width=800, # Adjusting width of the plot ) fig.update_xaxes( tickvals=[1000, 5000, 10000], # Set specific tick values ticktext=["1k", "5k", "10k"], # Set corresponding tick labels ) fig.update_traces( marker=dict(size=12, opacity=0.7) ) # Adjust marker size and opacity

# Show plot

fig.show()

# Save plot as HTML file

fig.write_html("performance.html") plot_performance(df)

Interesting. Let's try to summarize the insights we got from this experiment:

- The `feed_async_iterable` method is approximately 3x faster than the `feed_iterable` method for this specific setup.

- Note that this will vary depending on the network latency between the client and the Vespa instance.

- If you are feeding from a cloud instance with less latency to the Vespa instance, the difference between the methods will be less, and the `feed_iterable` method might even be faster.

- Still prefer to use the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli) if you *really* care about performance. 🚀

- If you want to use pyvespa, prefer the `feed_async_iterable`- method, if you are I/O-bound.

## Cleanup[¶](#cleanup)

In \[26\]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()

```
Deactivated vespa-team.feedperformancecloud in dev.aws-us-east-1c
Deleted instance vespa-team.feedperformancecloud.default
```

## Next steps[¶](#next-steps)

Check out some of the other [examples](https://vespa-engine.github.io/pyvespa/examples) in the documentation.
