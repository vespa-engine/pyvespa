# Read and write operations[¶](#read-and-write-operations)

This notebook documents ways to feed, get, update and delete data:

- Using context manager with `with` for efficiently managing resources
- Feeding streams of data using `feed_iter` which can feed from streams, Iterables, Lists and files by the use of generators

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

## Deploy a sample application[¶](#deploy-a-sample-application)

[Install pyvespa](https://pyvespa.readthedocs.io/) and start Docker, validate minimum 4G available:

In \[1\]:

Copied!

```
!docker info | grep "Total Memory"
```

!docker info | grep "Total Memory"

Define a simple application package with five fields

In \[1\]:

Copied!

```
from vespa.application import ApplicationPackage
from vespa.package import Schema, Document, Field, FieldSet, HNSW, RankProfile

app_package = ApplicationPackage(
    name="vector",
    schema=[
        Schema(
            name="doc",
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["attribute", "summary"]),
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
                    Field(
                        name="popularity",
                        type="float",
                        indexing=["attribute", "summary"],
                    ),
                    Field(
                        name="embedding",
                        type="tensor<bfloat16>(x[1536])",
                        indexing=["attribute", "summary", "index"],
                        ann=HNSW(
                            distance_metric="innerproduct",
                            max_links_per_node=16,
                            neighbors_to_explore_at_insert=128,
                        ),
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["title", "body"])],
            rank_profiles=[
                RankProfile(
                    name="default",
                    inputs=[("query(q)", "tensor<float>(x[1536])")],
                    first_phase="closeness(field, embedding)",
                )
            ],
        )
    ],
)
```

from vespa.application import ApplicationPackage from vespa.package import Schema, Document, Field, FieldSet, HNSW, RankProfile app_package = ApplicationPackage( name="vector", schema=\[ Schema( name="doc", document=Document( fields=\[ Field(name="id", type="string", indexing=["attribute", "summary"]), Field( name="title", type="string", indexing=["index", "summary"], index="enable-bm25", ), Field( name="body", type="string", indexing=["index", "summary"], index="enable-bm25", ), Field( name="popularity", type="float", indexing=["attribute", "summary"], ), Field( name="embedding", type="tensor<bfloat16>(x[1536])", indexing=["attribute", "summary", "index"], ann=HNSW( distance_metric="innerproduct", max_links_per_node=16, neighbors_to_explore_at_insert=128, ), ), \] ), fieldsets=\[FieldSet(name="default", fields=["title", "body"])\], rank_profiles=\[ RankProfile( name="default", inputs=\[("query(q)", "tensor<float>(x[1536])")\], first_phase="closeness(field, embedding)", ) \], ) \], )

In \[2\]:

Copied!

```
from vespa.deployment import VespaDocker

vespa_docker = VespaDocker()
app = vespa_docker.deploy(application_package=app_package)
```

from vespa.deployment import VespaDocker vespa_docker = VespaDocker() app = vespa_docker.deploy(application_package=app_package)

```
Waiting for configuration server, 0/60 seconds...
Waiting for configuration server, 5/60 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 0/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 5/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 10/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 15/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 20/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Application is up!
Finished deployment.
```

## Feed data by streaming over Iterable type[¶](#feed-data-by-streaming-over-iterable-type)

This example notebook uses the [dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) dataset (1M OpenAI Embeddings (1536 dimensions) from June 2023).

The `streaming=True` option allow paging the data on-demand from HF S3. This is extremely useful for large datasets, where the data does not fit in memory and downloading the entire dataset is not needed. Read more about [datasets stream](https://huggingface.co/docs/datasets/stream).

In \[ \]:

Copied!

```
from datasets import load_dataset

dataset = load_dataset(
    "KShivendu/dbpedia-entities-openai-1M", split="train", streaming=True
).take(1000)
```

from datasets import load_dataset dataset = load_dataset( "KShivendu/dbpedia-entities-openai-1M", split="train", streaming=True ).take(1000)

### Converting to dataset field names to Vespa schema field names[¶](#converting-to-dataset-field-names-to-vespa-schema-field-names)

We need to convert the dataset field names to the configured Vespa schema field names, we do this with a simple lambda function.

The map function does not page the data, the map step is performed lazily if we start iterating over the dataset. This allows chaining of map operations where the lambda is yielding the next document.

In \[4\]:

Copied!

```
pyvespa_feed_format = dataset.map(
    lambda x: {"id": x["_id"], "fields": {"id": x["_id"], "embedding": x["openai"]}}
)
```

pyvespa_feed_format = dataset.map( lambda x: {"id": x["\_id"], "fields": {"id": x["\_id"], "embedding": x["openai"]}} )

Feed using [feed_iterable](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa.feed_iterable) which accepts an `Iterable`. `feed_iterable` accepts a callback callable routine that is called for every single data operation so we can check the result. If the result `is_successful()` the operation is persisted and applied in Vespa.

In \[5\]:

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

In \[6\]:

Copied!

```
app.feed_iterable(
    iter=pyvespa_feed_format,
    schema="doc",
    namespace="benchmark",
    callback=callback,
    max_queue_size=4000,
    max_workers=16,
    max_connections=16,
)
```

app.feed_iterable( iter=pyvespa_feed_format, schema="doc", namespace="benchmark", callback=callback, max_queue_size=4000, max_workers=16, max_connections=16, )

### Feeding with generators[¶](#feeding-with-generators)

The above handled streaming data from a remote repo, we can also use generators or just List. In this example, we generate synthetic data using a generator function.

In \[7\]:

Copied!

```
def my_generator() -> dict:
    for i in range(1000):
        yield {
            "id": str(i),
            "fields": {
                "id": str(i),
                "title": "title",
                "body": "this is body",
                "popularity": 1.0,
            },
        }
```

def my_generator() -> dict: for i in range(1000): yield { "id": str(i), "fields": { "id": str(i), "title": "title", "body": "this is body", "popularity": 1.0, }, }

In \[8\]:

Copied!

```
app.feed_iterable(
    iter=my_generator(),
    schema="doc",
    namespace="benchmark",
    callback=callback,
    max_queue_size=4000,
    max_workers=12,
    max_connections=12,
)
```

app.feed_iterable( iter=my_generator(), schema="doc", namespace="benchmark", callback=callback, max_queue_size=4000, max_workers=12, max_connections=12, )

### Visiting[¶](#visiting)

Visiting is a feature to efficiently get or process a set of documents, identified by a document selection expression. Visit yields multiple slices (run concurrently) each yielding responses (depending on number of documents in each slice). This allows for custom handling of each response.

Visiting can be useful for exporting data, for example for ML training or for migrating a vespa application.

In \[9\]:

Copied!

```
all_docs = []
for slice in app.visit(
    content_cluster_name="vector_content",
    schema="doc",
    namespace="benchmark",
    selection="true",  # Document selection - see https://docs.vespa.ai/en/reference/document-select-language.html
    slices=4,
    wanted_document_count=300,
):
    for response in slice:
        print(response.number_documents_retrieved)
        all_docs.extend(response.documents)
```

all_docs = [] for slice in app.visit( content_cluster_name="vector_content", schema="doc", namespace="benchmark", selection="true", # Document selection - see https://docs.vespa.ai/en/reference/document-select-language.html slices=4, wanted_document_count=300, ): for response in slice: print(response.number_documents_retrieved) all_docs.extend(response.documents)

```
300
196
303
185
309
191
303
213
```

In \[10\]:

Copied!

```
len(all_docs)
```

len(all_docs)

Out\[10\]:

```
2000
```

In \[11\]:

Copied!

```
for slice in app.visit(
    content_cluster_name="vector_content", wanted_document_count=1000
):
    for response in slice:
        print(response.number_documents_retrieved)
```

for slice in app.visit( content_cluster_name="vector_content", wanted_document_count=1000 ): for response in slice: print(response.number_documents_retrieved)

```
190
189
226
205
184
214
202
181
217
192
```

### Updates[¶](#updates)

Using a similar generator we can update the fake data we added. This performs [partial updates](https://docs.vespa.ai/en/partial-updates.html), assigning the `popularity` field to have the value `2.0`.

In \[12\]:

Copied!

```
def my_update_generator() -> dict:
    for i in range(1000):
        yield {"id": str(i), "fields": {"popularity": 2.0}}
```

def my_update_generator() -> dict: for i in range(1000): yield {"id": str(i), "fields": {"popularity": 2.0}}

In \[13\]:

Copied!

```
app.feed_iterable(
    iter=my_update_generator(),
    schema="doc",
    namespace="benchmark",
    callback=callback,
    operation_type="update",
    max_queue_size=4000,
    max_workers=12,
    max_connections=12,
)
```

app.feed_iterable( iter=my_update_generator(), schema="doc", namespace="benchmark", callback=callback, operation_type="update", max_queue_size=4000, max_workers=12, max_connections=12, )

## Other update operations[¶](#other-update-operations)

We can also perform other update operations, see [Vespa docs on reads and writes](https://docs.vespa.ai/en/reads-and-writes.html). To achieve this we need to set the `auto_assign` parameter to `False` in the `feed_iterable` method (which will pass this to `update_data_point`-method).

In \[14\]:

Copied!

```
def my_increment_generator() -> dict:
    for i in range(1000):
        yield {"id": str(i), "fields": {"popularity": {"increment": 1.0}}}
```

def my_increment_generator() -> dict: for i in range(1000): yield {"id": str(i), "fields": {"popularity": {"increment": 1.0}}}

In \[15\]:

Copied!

```
app.feed_iterable(
    iter=my_increment_generator(),
    schema="doc",
    namespace="benchmark",
    callback=callback,
    operation_type="update",
    max_queue_size=4000,
    max_workers=12,
    max_connections=12,
    auto_assign=False,
)
```

app.feed_iterable( iter=my_increment_generator(), schema="doc", namespace="benchmark", callback=callback, operation_type="update", max_queue_size=4000, max_workers=12, max_connections=12, auto_assign=False, )

We can now query the data, notice how we use a context manager `with` to close connection after query This avoids resource leakage and allows for reuse of connections. In this case, we only do a single query and there is no need for having more than one connection. Setting more connections will just increase connection level overhead.

In \[16\]:

Copied!

```
from vespa.io import VespaQueryResponse

with app.syncio(connections=1):
    response: VespaQueryResponse = app.query(
        yql="select id from doc where popularity > 2.5", hits=0
    )
    print(response.number_documents_retrieved)
```

from vespa.io import VespaQueryResponse with app.syncio(connections=1): response: VespaQueryResponse = app.query( yql="select id from doc where popularity > 2.5", hits=0 ) print(response.number_documents_retrieved)

```
1000
```

### Deleting[¶](#deleting)

Delete all the synthetic data with a custom generator. Now we don't need the `fields` key.

In \[16\]:

Copied!

```
def my_delete_generator() -> dict:
    for i in range(1000):
        yield {"id": str(i)}


app.feed_iterable(
    iter=my_delete_generator(),
    schema="doc",
    namespace="benchmark",
    callback=callback,
    operation_type="delete",
    max_queue_size=5000,
    max_workers=48,
    max_connections=48,
)
```

def my_delete_generator() -> dict: for i in range(1000): yield {"id": str(i)} app.feed_iterable( iter=my_delete_generator(), schema="doc", namespace="benchmark", callback=callback, operation_type="delete", max_queue_size=5000, max_workers=48, max_connections=48, )

## Feeding operations from a file[¶](#feeding-operations-from-a-file)

This demonstrates how we can use `feed_iter` to feed from a large file without reading the entire file, this also uses a generator.

First we dump some operations to the file and peak at the first line:

In \[17\]:

Copied!

```
# Dump some operation to a jsonl file, we store it in the format expected by pyvespa
# This to demonstrate feeding from a file in the next section.
import json

with open("documents.jsonl", "w") as f:
    for doc in dataset:
        d = {"id": doc["_id"], "fields": {"id": doc["_id"], "embedding": doc["openai"]}}
        f.write(json.dumps(d) + "\n")
```

# Dump some operation to a jsonl file, we store it in the format expected by pyvespa

# This to demonstrate feeding from a file in the next section.

import json with open("documents.jsonl", "w") as f: for doc in dataset: d = {"id": doc["\_id"], "fields": {"id": doc["\_id"], "embedding": doc["openai"]}} f.write(json.dumps(d) + "\\n")

Define the file generator that will yield one line at a time

In \[18\]:

Copied!

```
import json


def from_file_generator() -> dict:
    with open("documents.jsonl") as f:
        for line in f:
            yield json.loads(line)
```

import json def from_file_generator() -> dict: with open("documents.jsonl") as f: for line in f: yield json.loads(line)

In \[19\]:

Copied!

```
app.feed_iterable(
    iter=from_file_generator(),
    schema="doc",
    namespace="benchmark",
    callback=callback,
    operation_type="feed",
    max_queue_size=4000,
    max_workers=32,
    max_connections=32,
)
```

app.feed_iterable( iter=from_file_generator(), schema="doc", namespace="benchmark", callback=callback, operation_type="feed", max_queue_size=4000, max_workers=32, max_connections=32, )

### Get and Feed individual data points[¶](#get-and-feed-individual-data-points)

Feed a single data point to Vespa

In \[20\]:

Copied!

```
with app.syncio(connections=1):
    response: VespaResponse = app.feed_data_point(
        schema="doc",
        namespace="benchmark",
        data_id="1",
        fields={
            "id": "1",
            "title": "title",
            "body": "this is body",
            "popularity": 1.0,
        },
    )
    print(response.is_successful())
    print(response.get_json())
```

with app.syncio(connections=1): response: VespaResponse = app.feed_data_point( schema="doc", namespace="benchmark", data_id="1", fields={ "id": "1", "title": "title", "body": "this is body", "popularity": 1.0, }, ) print(response.is_successful()) print(response.get_json())

```
True
{'pathId': '/document/v1/benchmark/doc/docid/1', 'id': 'id:benchmark:doc::1'}
```

Get the same document, try also to change data_id to a document that does not exist which will raise a 404 http error.

In \[21\]:

Copied!

```
with app.syncio(connections=1):
    response: VespaResponse = app.get_data(
        schema="doc",
        namespace="benchmark",
        data_id="1",
    )
    print(response.is_successful())
    print(response.get_json())
```

with app.syncio(connections=1): response: VespaResponse = app.get_data( schema="doc", namespace="benchmark", data_id="1", ) print(response.is_successful()) print(response.get_json())

```
True
{'pathId': '/document/v1/benchmark/doc/docid/1', 'id': 'id:benchmark:doc::1', 'fields': {'body': 'this is body', 'title': 'title', 'popularity': 1.0, 'id': '1'}}
```

### Upsert[¶](#upsert)

The following sends an update operation, if the document exist, the popularity field will be updated to take the value 3.0, and if the document does not exist, it's created and where the popularity value is 3.0.

In \[22\]:

Copied!

```
with app.syncio(connections=1):
    response: VespaResponse = app.update_data(
        schema="doc",
        namespace="benchmark",
        data_id="does-not-exist",
        fields={"popularity": 3.0},
        create=True,
    )
    print(response.is_successful())
    print(response.get_json())
```

with app.syncio(connections=1): response: VespaResponse = app.update_data( schema="doc", namespace="benchmark", data_id="does-not-exist", fields={"popularity": 3.0}, create=True, ) print(response.is_successful()) print(response.get_json())

```
True
{'pathId': '/document/v1/benchmark/doc/docid/does-not-exist', 'id': 'id:benchmark:doc::does-not-exist'}
```

In \[23\]:

Copied!

```
with app.syncio(connections=1):
    response: VespaResponse = app.get_data(
        schema="doc",
        namespace="benchmark",
        data_id="does-not-exist",
    )
    print(response.is_successful())
    print(response.get_json())
```

with app.syncio(connections=1): response: VespaResponse = app.get_data( schema="doc", namespace="benchmark", data_id="does-not-exist", ) print(response.is_successful()) print(response.get_json())

```
True
{'pathId': '/document/v1/benchmark/doc/docid/does-not-exist', 'id': 'id:benchmark:doc::does-not-exist', 'fields': {'popularity': 3.0}}
```

## Cleanup[¶](#cleanup)

In \[24\]:

Copied!

```
vespa_docker.container.stop()
vespa_docker.container.remove()
```

vespa_docker.container.stop() vespa_docker.container.remove()

## Next steps[¶](#next-steps)

Read more on writing to Vespa in [reads-and-writes](https://docs.vespa.ai/en/reads-and-writes.html).
