# Advanced Configuration[¶](#advanced-configuration)

This notebook demonstrates how to use pyvespa's advanced configuration features to customize Vespa applications beyond the basic settings. You'll learn to express Vespa's XML configuration files using Python code for greater flexibility and control.

## What you'll learn[¶](#what-youll-learn)

1. **[services.xml Configuration](#services-xml-configuration)** - Configure `services.xml` using the `ServicesConfiguration` object to customize system behavior (document expiry, threading, tuning parameters). Available since `pyvespa=0.50.0`
1. **[Query Profiles Configuration](#query-profiles-configuration)** - Define multiple query profiles and query profile types programmatically using the new configuration approach. Available since `pyvespa=0.60.0`
1. **[deployment.xml Configuration](#deploymentxml-configuration)** - Configure deployment zones, regions and windows to block upgrades. Applicable for Vespa Cloud only. Available since `pyvespa=0.60.0`

## Why?[¶](#why)

pyvespa has proven to be a preferred framework for deploying and managing Vespa applications. With the legacy configuration methods, not all possible configurations were available. The new approach ensures full feature parity with the XML configuration options.

## Configuration Approach[¶](#configuration-approach)

The `vespa.configuration` modules in pyvespa provides a **Vespa Tag (VT)** system that mirrors Vespa's XML configuration structure:

- **Tags**: Python functions representing XML elements (e.g., `container()`, `content()`, `query_profile()`)
- **Attributes**: Function parameters that become XML attributes (hyphens become underscores: `garbage-collection` → `garbage_collection`)
- **Values**: Automatic type conversion and XML escaping (no manual escaping needed)
- **Structure**: Nested function calls create the XML hierarchy

**Example**: This Python code:

```
service_config = ServicesConfiguration(
  name="myapp",
  container(id="myapp_container", version="1.0")(
      search(),
      document_api()
  )
)
service_config.to_xml()
```

Generates this XML:

```
<services>
  <container id="myapp_container" version="1.0">
    <search></search>
    <document-api></document-api>
  </container>
</services>
```

## Prerequisites[¶](#prerequisites)

- pyvespa installed and Docker running with at least 6GB memory
- Understanding of basic Vespa concepts (schemas, deployment)

For detailed XML configuration options, refer to:

- [Vespa services.xml reference](https://docs.vespa.ai/en/reference/services.html)
- [Query profiles reference](https://docs.vespa.ai/en/querying/query-profiles.html)
- [Deployment reference](https://docs.vespa.ai/en/reference/deployment.html)

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

[Install pyvespa](https://pyvespa.readthedocs.io/) and start Docker Daemon, validate minimum 6G available:

In \[1\]:

Copied!

```
#!pip3 install pyvespa
#!docker info | grep "Total Memory"
```

#!pip3 install pyvespa #!docker info | grep "Total Memory"

## services.xml Configuration[¶](#servicesxml-configuration)

### Example 1 - Configure document-expiry[¶](#example-1-configure-document-expiry)

As an example of a common use case for advanced configuration, we will configure document-expiry. This feature allows you to set a time-to-live for documents in your Vespa application. This is useful when you have documents that are only relevant for a certain period of time, and you want to avoid serving stale data.

For reference, see the [docs on document-expiry](https://docs.vespa.ai/en/documents.html#document-expiry).

#### Define a schema[¶](#define-a-schema)

We define a simple schema, with a timestamp field that we will use in the document selection expression to set the document-expiry.

Note that the fields that are referenced in the selection expression should be attributes(in-memory).

Also, either the fields should be set with `fast-access` or the number of searchable copies in the content cluster should be the same as the redundancy. Otherwise, the document selection maintenance will be slow and have a major performance impact on the system.

In \[2\]:

Copied!

```
from vespa.package import Document, Field, Schema, ApplicationPackage

application_name = "music"
music_schema = Schema(
    name=application_name,
    document=Document(
        fields=[
            Field(
                name="artist",
                type="string",
                indexing=["attribute", "summary"],
            ),
            Field(
                name="title",
                type="string",
                indexing=["attribute", "summary"],
            ),
            Field(
                name="timestamp",
                type="long",
                indexing=["attribute", "summary"],
                attribute=["fast-access"],
            ),
        ]
    ),
)
```

from vespa.package import Document, Field, Schema, ApplicationPackage application_name = "music" music_schema = Schema( name=application_name, document=Document( fields=\[ Field( name="artist", type="string", indexing=["attribute", "summary"], ), Field( name="title", type="string", indexing=["attribute", "summary"], ), Field( name="timestamp", type="long", indexing=["attribute", "summary"], attribute=["fast-access"], ), \] ), )

### The `ServicesConfiguration` object[¶](#the-servicesconfiguration-object)

The `ServicesConfiguration` object allows you to define any configuration you want in the `services.xml` file.

The syntax is as follows:

In \[3\]:

Copied!

```
from vespa.package import ServicesConfiguration
from vespa.configuration.services import (
    services,
    container,
    search,
    document_api,
    document_processing,
    content,
    redundancy,
    documents,
    document,
    node,
    nodes,
)

# Create a ServicesConfiguration with document-expiry set to 1 day (timestamp > now() - 86400)
services_config = ServicesConfiguration(
    application_name=application_name,
    services_config=services(
        container(
            search(),
            document_api(),
            document_processing(),
            id=f"{application_name}_container",
            version="1.0",
        ),
        content(
            redundancy("1"),
            documents(
                document(
                    type=application_name,
                    mode="index",
                    # Note that the selection-expression does not need to be escaped, as it will be automatically escaped during xml-serialization
                    selection="music.timestamp > now() - 86400",
                ),
                garbage_collection="true",
            ),
            nodes(node(distribution_key="0", hostalias="node1")),
            id=f"{application_name}_content",
            version="1.0",
        ),
    ),
)
application_package = ApplicationPackage(
    name=application_name,
    schema=[music_schema],
    services_config=services_config,
)
```

from vespa.package import ServicesConfiguration from vespa.configuration.services import ( services, container, search, document_api, document_processing, content, redundancy, documents, document, node, nodes, )

# Create a ServicesConfiguration with document-expiry set to 1 day (timestamp > now() - 86400)

services_config = ServicesConfiguration( application_name=application_name, services_config=services( container( search(), document_api(), document_processing(), id=f"{application_name}\_container", version="1.0", ), content( redundancy("1"), documents( document( type=application_name, mode="index",

# Note that the selection-expression does not need to be escaped, as it will be automatically escaped during xml-serialization

selection="music.timestamp > now() - 86400", ), garbage_collection="true", ), nodes(node(distribution_key="0", hostalias="node1")), id=f"{application_name}\_content", version="1.0", ), ), ) application_package = ApplicationPackage( name=application_name, schema=[music_schema], services_config=services_config, )

There are some useful gotchas to keep in mind when constructing the `ServicesConfiguration` object.

First, let's establish a common vocabulary through an example. Consider the following `services.xml` file, which is what we are actually representing with the `ServicesConfiguration` object from the previous cell:

```
<?xml version="1.0" encoding="UTF-8" ?>
<services>
  <container id="music_container" version="1.0">
    <search></search>
    <document-api></document-api>
    <document-processing></document-processing>
  </container>
  <content id="music_content" version="1.0">
    <redundancy>1</redundancy>
    <documents garbage-collection="true">
      <document type="music" mode="index" selection="music.timestamp &gt; now() - 86400"></document>
    </documents>
    <nodes>
      <node distribution-key="0" hostalias="node1"></node>
    </nodes>
  </content>
</services>
```

In this example, `services`, `container`, `search`, `document-api`, `document-processing`, `content`, `redundancy`, `documents`, `document`, and `nodes` are *tags*. The `id`, `version`, `type`, `mode`, `selection`, `distribution-key`, `hostalias`, and `garbage-collection` are *attributes*, with a corresponding *value*.

### Tag names[¶](#tag-names)

All tags as referenced in the [Vespa documentation](https://docs.vespa.ai/en/reference/services.html) are available in `vespa.configuration.{services,query_profiles,deployment}` modules with the following modifications:

- All `-` in the tag names are replaced by `_` to avoid conflicts with Python syntax.
- Some tags that are Python reserved words (or commonly used objects) are constructed by adding a `_` at the end of the tag name. To see which tags are affected, you can check this variable:

In \[4\]:

Copied!

```
from vespa.configuration.vt import replace_reserved

replace_reserved
```

from vespa.configuration.vt import replace_reserved replace_reserved

Out\[4\]:

```
{'type': 'type_',
 'class': 'class_',
 'for': 'for_',
 'time': 'time_',
 'io': 'io_',
 'from': 'from_',
 'match': 'match_'}
```

Only valid tags will be exported by the `vespa.configuration.` modules.

### Attributes[¶](#attributes)

- *any* attribute can be passed to the tag constructor (no validation at construction time).
- The attribute name should be the same as in the Vespa documentation, but with `-` replaced by `_`. For example, the `garbage-collection` attribute in the `query` tag should be passed as `garbage_collection`.
- In case the attribute name is a Python reserved word, the same rule as for the tag names applies (add `_` at the end). An example of this is the `global` attribute which should be passed as `global_`.
- Some attributes, such as `id`, in the `container` tag, are mandatory and should be passed as positional arguments to the tag constructor.

### Values[¶](#values)

- The value of an attribute can be a string, an integer, or a boolean. For types `bool` and `int`, the value is converted to a string (lowercased for `bool`). If you need to pass a float, you should convert it to a string before passing it to the tag constructor, e.g. `container(version="1.0")`.
- Note that we are *not* escaping the values. In the xml file, the value of the `selection` attribute in the `document` tag is `music.timestamp &gt; now() - 86400`. (`&gt;` is the escaped form of `>`.) When passing this value to the `document` tag constructor in python, we should *not* escape the `>` character, i.e. `document(selection="music.timestamp > now() - 86400")`.

## Deploy the Vespa application[¶](#deploy-the-vespa-application)

Deploy `package` on the local machine using Docker, without leaving the notebook, by creating an instance of [VespaDocker](https://vespa-engine.github.io/pyvespa/api/vespa/deployment.md#vespa.deployment.VespaDocker). `VespaDocker` connects to the local Docker daemon socket and starts the [Vespa docker image](https://hub.docker.com/r/vespaengine/vespa/).

If this step fails, please check that the Docker daemon is running, and that the Docker daemon socket can be used by clients (Configurable under advanced settings in Docker Desktop).

In \[5\]:

Copied!

```
from vespa.deployment import VespaDocker

vespa_docker = VespaDocker()
app = vespa_docker.deploy(application_package=application_package)
```

from vespa.deployment import VespaDocker vespa_docker = VespaDocker() app = vespa_docker.deploy(application_package=application_package)

```
Waiting for configuration server, 0/60 seconds...
Waiting for application to come up, 0/300 seconds.
Waiting for application to come up, 5/300 seconds.
Waiting for application to come up, 10/300 seconds.
Application is up!
Finished deployment.
```

`app` now holds a reference to a [Vespa](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa) instance. see this [notebook](https://vespa-engine.github.io/pyvespa/authenticating-to-vespa-cloud.md) for details on authenticating to Vespa Cloud.

## Feeding documents to Vespa[¶](#feeding-documents-to-vespa)

Now, let us feed some documents to Vespa. We will feed one document with a timestamp of 24 hours (+1 sec (86401)) ago and another document with a timestamp of the current time. We will then query the documents to check verify that the document-expiry is working as expected.

In \[6\]:

Copied!

```
import time

docs_to_feed = [
    {
        "id": "1",
        "fields": {
            "artist": "Snoop Dogg",
            "title": "Gin and Juice",
            "timestamp": int(time.time()) - 86401,
        },
    },
    {
        "id": "2",
        "fields": {
            "artist": "Dr.Dre",
            "title": "Still D.R.E",
            "timestamp": int(time.time()),
        },
    },
]
```

import time docs_to_feed = [ { "id": "1", "fields": { "artist": "Snoop Dogg", "title": "Gin and Juice", "timestamp": int(time.time()) - 86401, }, }, { "id": "2", "fields": { "artist": "Dr.Dre", "title": "Still D.R.E", "timestamp": int(time.time()), }, }, ]

In \[7\]:

Copied!

```
from vespa.io import VespaResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Error when feeding document {id}: {response.get_json()}")


app.feed_iterable(docs_to_feed, schema=application_name, callback=callback)
```

from vespa.io import VespaResponse def callback(response: VespaResponse, id: str): if not response.is_successful(): print(f"Error when feeding document {id}: {response.get_json()}") app.feed_iterable(docs_to_feed, schema=application_name, callback=callback)

## Verify document expiry through visiting[¶](#verify-document-expiry-through-visiting)

[Visiting](https://docs.vespa.ai/en/visiting.html) is a feature to efficiently get or process a set of documents, identified by a [document selection](https://docs.vespa.ai/en/reference/document-select-language.html) expression. Here is how you can use visiting in pyvespa:

In \[8\]:

Copied!

```
visit_results = []
for slice_ in app.visit(
    schema=application_name,
    content_cluster_name=f"{application_name}_content",
    timeout="5s",
):
    for response in slice_:
        visit_results.append(response.json)
visit_results
```

visit_results = [] for slice\_ in app.visit( schema=application_name, content_cluster_name=f"{application_name}_content", timeout="5s", ): for response in slice_: visit_results.append(response.json) visit_results

Out\[8\]:

```
[{'pathId': '/document/v1/music/music/docid/',
  'documents': [{'id': 'id:music:music::2',
    'fields': {'artist': 'Dr.Dre',
     'title': 'Still D.R.E',
     'timestamp': 1754981413}}],
  'documentCount': 1}]
```

We can see that the document with the timestamp of 24 hours ago is not returned by the query, while the document with the current timestamp is returned.

### Clean up[¶](#clean-up)

In \[9\]:

Copied!

```
vespa_docker.container.stop()
vespa_docker.container.remove()
```

vespa_docker.container.stop() vespa_docker.container.remove()

### Example 2 - Configuring `requestthreads` per search[¶](#example-2-configuring-requestthreads-per-search)

In Vespa, there are several configuration options that might be tuned to optimize the serving latency of your application. For an overview, see the [Vespa documentation - Vespa Serving Scaling Guide](https://docs.vespa.ai/en/performance/sizing-search.html). An example of a configuration that one might want to tune is the `requestthreads` `persearch` [parameter](https://docs.vespa.ai/en/reference/services-content.html#requestthreads). This parameter controls the number of search threads that are used to handle each search on the content nodes. The default value is 1.

For some applications, where a significant portion of the work per query is linear with the number of documents, increasing the number of `requestthreads` `persearch` can improve the serving latency, as it allows more parallelism in the search phase.

Examples of potentially expensive work that scales linearly with the number of documents, and thus are likely to benefit from increasing `requestthreads` `persearch` are: - Xgboost inference with a large GDBT-model - ONNX inference, e.g with a crossencoder. - MaxSim-operations for late interaction scoring, as in ColBERT and ColPali. - Exact nearest neighbor search.

Example of query operators that are less likely to benefit from increasing `requestthreads` `persearch` are: - `wand`/`weakAnd`, see [Using wand with Vespa](https://docs.vespa.ai/en/using-wand-with-vespa.html). - Approximate nearest neighbor search with HNSW.

In this example, we will demonstrate an example of configuring `requestthreads` `persearch` to 4 for an application where a Crossencoder is used in first-phase ranking. The demo is based on the [Cross-encoders for global reranking](https://vespa-engine.github.io/pyvespa/examples/cross-encoders-for-global-reranking.md) guide, but here we will use a cross-encoder in first-phase instead of global-phase. First-phase and second-phase ranking are executed on the content nodes, while global-phase ranking is executed on the container node. See [Phased ranking](https://docs.vespa.ai/en/phased-ranking.html) for more details.

### Download the crossencoder-model[¶](#download-the-crossencoder-model)

In \[10\]:

Copied!

```
from pathlib import Path
import requests
from vespa.deployment import VespaDocker

# Download the model if it doesn't exist
url = "https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1/resolve/main/onnx/model.onnx"
local_model_path = "model/model.onnx"
if not Path(local_model_path).exists():
    print("Downloading the mxbai-rerank model...")
    r = requests.get(url)
    Path(local_model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_model_path, "wb") as f:
        f.write(r.content)
        print(f"Downloaded model to {local_model_path}")
else:
    print("Model already exists, skipping download.")
```

from pathlib import Path import requests from vespa.deployment import VespaDocker

# Download the model if it doesn't exist

url = "https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1/resolve/main/onnx/model.onnx" local_model_path = "model/model.onnx" if not Path(local_model_path).exists(): print("Downloading the mxbai-rerank model...") r = requests.get(url) Path(local_model_path).parent.mkdir(parents=True, exist_ok=True) with open(local_model_path, "wb") as f: f.write(r.content) print(f"Downloaded model to {local_model_path}") else: print("Model already exists, skipping download.")

```
Model already exists, skipping download.
```

### Define a schema[¶](#define-a-schema)

In \[11\]:

Copied!

```
from vespa.package import (
    OnnxModel,
    RankProfile,
    Schema,
    ApplicationPackage,
    Field,
    FieldSet,
    Function,
    FirstPhaseRanking,
    Document,
)


application_name = "requestthreads"

# Define the reranking, as we will use it for two different rank profiles
reranking = FirstPhaseRanking(
    keep_rank_count=8,
    expression="sigmoid(onnx(crossencoder).logits{d0:0,d1:0})",
)

# Define the schema
schema = Schema(
    name="doc",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary", "attribute"]),
            Field(
                name="text",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="body_tokens",
                type="tensor<float>(d0[512])",
                indexing=[
                    "input text",
                    "embed tokenizer",
                    "attribute",
                    "summary",
                ],
                is_document_field=False,  # Indicates a synthetic field
            ),
        ],
    ),
    fieldsets=[FieldSet(name="default", fields=["text"])],
    models=[
        OnnxModel(
            model_name="crossencoder",
            model_file_path=f"{local_model_path}",
            inputs={
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
            },
            outputs={"logits": "logits"},
        )
    ],
    rank_profiles=[
        RankProfile(name="bm25", first_phase="bm25(text)"),
        RankProfile(
            name="reranking",
            inherits="default",
            inputs=[("query(q)", "tensor<float>(d0[64])")],
            functions=[
                Function(
                    name="input_ids",
                    expression="customTokenInputIds(1, 2, 512, query(q), attribute(body_tokens))",
                ),
                Function(
                    name="attention_mask",
                    expression="tokenAttentionMask(512, query(q), attribute(body_tokens))",
                ),
            ],
            first_phase=reranking,
            summary_features=[
                "query(q)",
                "input_ids",
                "attention_mask",
                "onnx(crossencoder).logits",
            ],
        ),
        RankProfile(
            name="one-thread-profile",
            first_phase=reranking,
            inherits="reranking",
            num_threads_per_search=1,
        ),
    ],
)
```

from vespa.package import ( OnnxModel, RankProfile, Schema, ApplicationPackage, Field, FieldSet, Function, FirstPhaseRanking, Document, ) application_name = "requestthreads"

# Define the reranking, as we will use it for two different rank profiles

reranking = FirstPhaseRanking( keep_rank_count=8, expression="sigmoid(onnx(crossencoder).logits{d0:0,d1:0})", )

# Define the schema

schema = Schema( name="doc", document=Document( fields=\[ Field(name="id", type="string", indexing=["summary", "attribute"]), Field( name="text", type="string", indexing=["index", "summary"], index="enable-bm25", ), Field( name="body_tokens", type="tensor<float>(d0[512])", indexing=[ "input text", "embed tokenizer", "attribute", "summary", ], is_document_field=False, # Indicates a synthetic field ), \], ), fieldsets=\[FieldSet(name="default", fields=["text"])\], models=[ OnnxModel( model_name="crossencoder", model_file_path=f"{local_model_path}", inputs={ "input_ids": "input_ids", "attention_mask": "attention_mask", }, outputs={"logits": "logits"}, ) ], rank_profiles=\[ RankProfile(name="bm25", first_phase="bm25(text)"), RankProfile( name="reranking", inherits="default", inputs=\[("query(q)", "tensor<float>(d0[64])")\], functions=[ Function( name="input_ids", expression="customTokenInputIds(1, 2, 512, query(q), attribute(body_tokens))", ), Function( name="attention_mask", expression="tokenAttentionMask(512, query(q), attribute(body_tokens))", ), ], first_phase=reranking, summary_features=[ "query(q)", "input_ids", "attention_mask", "onnx(crossencoder).logits", ], ), RankProfile( name="one-thread-profile", first_phase=reranking, inherits="reranking", num_threads_per_search=1, ), \], )

### Define the ServicesConfiguration[¶](#define-the-servicesconfiguration)

Note that the ServicesConfiguration may be used to define any configuration in the `services.xml` file. In this example, we are only configuring the `requestthreads` `persearch` parameter, but you can use the same approach to configure any other parameter.

For a full reference of the available configuration options, see the [Vespa documentation - services.xml](https://docs.vespa.ai/en/reference/services.html).

In \[12\]:

Copied!

```
from vespa.configuration.services import *
from vespa.package import ServicesConfiguration

# Define services configuration with persearch threads set to 4
services_config = ServicesConfiguration(
    application_name=f"{application_name}",
    services_config=services(
        container(id=f"{application_name}_default", version="1.0")(
            component(
                model(
                    url="https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1/raw/main/tokenizer.json"
                ),
                id="tokenizer",
                type="hugging-face-tokenizer",
            ),
            document_api(),
            search(),
        ),
        content(id=f"{application_name}", version="1.0")(
            min_redundancy("1"),
            documents(document(type="doc", mode="index")),
            engine(
                proton(
                    tuning(
                        searchnode(requestthreads(persearch("4"))),
                    ),
                ),
            ),
        ),
        version="1.0",
        minimum_required_vespa_version="8.311.28",
    ),
)
```

from vespa.configuration.services import * from vespa.package import ServicesConfiguration

# Define services configuration with persearch threads set to 4

services_config = ServicesConfiguration( application_name=f"{application_name}", services_config=services( container(id=f"{application_name}\_default", version="1.0")( component( model( url="https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1/raw/main/tokenizer.json" ), id="tokenizer", type="hugging-face-tokenizer", ), document_api(), search(), ), content(id=f"{application_name}", version="1.0")( min_redundancy("1"), documents(document(type="doc", mode="index")), engine( proton( tuning( searchnode(requestthreads(persearch("4"))), ), ), ), ), version="1.0", minimum_required_vespa_version="8.311.28", ), )

Now, we are ready to deploy our application-package with the defined `ServicesConfiguration`.

### Deploy the application package[¶](#deploy-the-application-package)

In \[13\]:

Copied!

```
app_package = ApplicationPackage(
    name=f"{application_name}",
    schema=[schema],
    services_config=services_config,
)
```

app_package = ApplicationPackage( name=f"{application_name}", schema=[schema], services_config=services_config, )

In \[14\]:

Copied!

```
vespa_docker = VespaDocker(port=8089)
app = vespa_docker.deploy(application_package=app_package)
```

vespa_docker = VespaDocker(port=8089) app = vespa_docker.deploy(application_package=app_package)

```
Waiting for configuration server, 0/60 seconds...
Waiting for application to come up, 0/300 seconds.
Waiting for application to come up, 5/300 seconds.
Waiting for application to come up, 10/300 seconds.
Application is up!
Finished deployment.
```

### Feed some sample documents[¶](#feed-some-sample-documents)

In \[15\]:

Copied!

```
sample_docs = [
    {"id": i, "fields": {"text": text}}
    for i, text in enumerate(
        [
            "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature. The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird'. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
            "was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961. Jane Austen was an English novelist known primarily for her six major novels, ",
            "which interpret, critique and comment upon the British landed gentry at the end of the 18th century. The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, ",
            "is among the most popular and critically acclaimed books of the modern era. 'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan.",
        ]
    )
]
app.feed_iterable(sample_docs, schema="doc")

# Define the query body
query_body = {
    "yql": "select * from sources * where userQuery();",
    "query": "who wrote to kill a mockingbird?",
    "timeout": "10s",
    "input.query(q)": "embed(tokenizer, @query)",
    "presentation.timing": "true",
}

# Warm-up query
app.query(body=query_body)
query_body_reranking = {
    **query_body,
    "ranking.profile": "reranking",
}
# Query with default persearch threads (set to 4)
with app.syncio() as sess:
    response_default = app.query(body=query_body_reranking)

# Query with num-threads-per-search overridden to 1
query_body_one_thread = {
    **query_body,
    "ranking.profile": "one-thread-profile",
    # "ranking.matching.numThreadsPerSearch": 1, Could potentiall also set numThreadsPerSearch in query parameters.
}
with app.syncio() as sess:
    response_one_thread = sess.query(body=query_body_one_thread)

# Extract query times
timing_default = response_default.json["timing"]["querytime"]
timing_one_thread = response_one_thread.json["timing"]["querytime"]
# Beautifully formatted statement of - num threads and ratio of query times
print(f"Query time with 4 threads: {timing_default:.2f}s")
print(f"Query time with 1 thread: {timing_one_thread:.2f}s")
ratio = timing_one_thread / timing_default
print(f"4 threads is approximately {ratio:.2f}x faster than 1 thread")
```

sample_docs = \[ {"id": i, "fields": {"text": text}} for i, text in enumerate( [ "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature. The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird'. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.", "was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961. Jane Austen was an English novelist known primarily for her six major novels, ", "which interpret, critique and comment upon the British landed gentry at the end of the 18th century. The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, ", "is among the most popular and critically acclaimed books of the modern era. 'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan.", ] ) \] app.feed_iterable(sample_docs, schema="doc")

# Define the query body

query_body = { "yql": "select * from sources * where userQuery();", "query": "who wrote to kill a mockingbird?", "timeout": "10s", "input.query(q)": "embed(tokenizer, @query)", "presentation.timing": "true", }

# Warm-up query

app.query(body=query_body) query_body_reranking = { \*\*query_body, "ranking.profile": "reranking", }

# Query with default persearch threads (set to 4)

with app.syncio() as sess: response_default = app.query(body=query_body_reranking)

# Query with num-threads-per-search overridden to 1

query_body_one_thread = { \*\*query_body, "ranking.profile": "one-thread-profile",

# "ranking.matching.numThreadsPerSearch": 1, Could potentiall also set numThreadsPerSearch in query parameters.

} with app.syncio() as sess: response_one_thread = sess.query(body=query_body_one_thread)

# Extract query times

timing_default = response_default.json["timing"]["querytime"] timing_one_thread = response_one_thread.json["timing"]["querytime"]

# Beautifully formatted statement of - num threads and ratio of query times

print(f"Query time with 4 threads: {timing_default:.2f}s") print(f"Query time with 1 thread: {timing_one_thread:.2f}s") ratio = timing_one_thread / timing_default print(f"4 threads is approximately {ratio:.2f}x faster than 1 thread")

```
Query time with 4 threads: 0.73s
Query time with 1 thread: 1.24s
4 threads is approximately 1.69x faster than 1 thread
```

## Query-profile Configuration[¶](#query-profile-configuration)

Until pyvespa version 0.60.0, this was the way to add a query profile or query profile type to the application package:

In \[16\]:

Copied!

```
from vespa.package import (
    QueryProfile,
    QueryProfileType,
    QueryTypeField,
    QueryField,
)

app_package = ApplicationPackage(
    name=f"{application_name}",
    schema=[music_schema],
    query_profile=QueryProfile(
        fields=[
            QueryField(
                name="hits",
                value="30",
            )
        ]
    ),
    query_profile_type=QueryProfileType(
        fields=[
            QueryTypeField(
                name="ranking.features.query(query_embedding)",
                type="tensor<float>(x[512])",
            )
        ]
    ),
)
```

from vespa.package import ( QueryProfile, QueryProfileType, QueryTypeField, QueryField, ) app_package = ApplicationPackage( name=f"{application_name}", schema=[music_schema], query_profile=QueryProfile( fields=[ QueryField( name="hits", value="30", ) ] ), query_profile_type=QueryProfileType( fields=\[ QueryTypeField( name="ranking.features.query(query_embedding)", type="tensor<float>(x[512])", ) \] ), )

As you can see from the reference in the [Vespa Docs](https://docs.vespa.ai/en/querying/query-profiles.html), this makes it impossible to define multiple query profiles or query profile types in the application package, and there are many variants you are unable to express.

## Query-profiles - new approach[¶](#query-profiles-new-approach)

By importing the tag-functions like this: `from vespa.configuration.query_profiles import *`, you can access all supported tags of a query profile or query profile type.

Pass these (one or as many as you like) to the `query_profile_config` parameter of your `ApplicationPackage`, and they will be added to the application package as query profiles or query profile types.

Only two validations are done at construction time:

1. The `id` attribute is mandatory for both query profiles and query profile types, as it is used to create the file name in the application package.
1. The top-level tag of each element in the `query_profile_config` list should be either `query_profile` or `query_profile_type`.

By using the new `query_profile_config` parameter, you can now express any combination of query profile or query profile type in python code, and add it to your `ApplicationPackage`.

Here are some examples:

In \[17\]:

Copied!

```
from vespa.configuration.query_profiles import *

# From https://docs.vespa.ai/en/tutorials/rag-blueprint.html#training-a-first-phase-ranking-model
qp_hybrid = query_profile(
    field("doc", name="schema"),
    field("embed(@query)", name="ranking.features.query(embedding)"),
    field("embed(@query)", name="ranking.features.query(float_embedding)"),
    field(-7.798639, name="ranking.features.query(intercept)"),
    field(
        13.383840,
        name="ranking.features.query(avg_top_3_chunk_sim_scores_param)",
    ),
    field(
        0.203145,
        name="ranking.features.query(avg_top_3_chunk_text_scores_param)",
    ),
    field(0.159914, name="ranking.features.query(bm25_chunks_param)"),
    field(0.191867, name="ranking.features.query(bm25_title_param)"),
    field(10.067169, name="ranking.features.query(max_chunk_sim_scores_param)"),
    field(0.153392, name="ranking.features.query(max_chunk_text_scores_param)"),
    field(
        """select *
        from %{schema}
        where userInput(@query) or
        ({label:"title_label", targetHits:100}nearestNeighbor(title_embedding, embedding)) or
        ({label:"chunks_label", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))""",
        name="yql",
    ),
    field(10, name="hits"),
    field("learned-linear", name="ranking.profile"),
    field("top_3_chunks", name="presentation.summary"),
    id="hybrid",
    type="hybrid-type",
)

qpt_hybrid = query_profile_type(
    field(
        name="ranking.features.query(embedding)",
        type="tensor<int8>(x[96])",
        mandatory=True,
        strict=True,
    ),
    field(
        name="ranking.features.query(float_embedding)",
        type="tensor<float>(x[384])",
        mandatory=True,
        strict=True,
    ),
    id="hybrid-type",
)
```

from vespa.configuration.query_profiles import \*

# From https://docs.vespa.ai/en/tutorials/rag-blueprint.html#training-a-first-phase-ranking-model

qp_hybrid = query_profile( field("doc", name="schema"), field("embed(@query)", name="ranking.features.query(embedding)"), field("embed(@query)", name="ranking.features.query(float_embedding)"), field(-7.798639, name="ranking.features.query(intercept)"), field( 13.383840, name="ranking.features.query(avg_top_3_chunk_sim_scores_param)", ), field( 0.203145, name="ranking.features.query(avg_top_3_chunk_text_scores_param)", ), field(0.159914, name="ranking.features.query(bm25_chunks_param)"), field(0.191867, name="ranking.features.query(bm25_title_param)"), field(10.067169, name="ranking.features.query(max_chunk_sim_scores_param)"), field(0.153392, name="ranking.features.query(max_chunk_text_scores_param)"), field( """select * from %{schema} where userInput(@query) or ({label:"title_label", targetHits:100}nearestNeighbor(title_embedding, embedding)) or ({label:"chunks_label", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))""", name="yql", ), field(10, name="hits"), field("learned-linear", name="ranking.profile"), field("top_3_chunks", name="presentation.summary"), id="hybrid", type="hybrid-type", ) qpt_hybrid = query_profile_type( field( name="ranking.features.query(embedding)", type="tensor<int8>(x[96])", mandatory=True, strict=True, ), field( name="ranking.features.query(float_embedding)", type="tensor<float>(x[384])", mandatory=True, strict=True, ), id="hybrid-type", )

As you can see below, we get type conversion (`True` -> `true`), XML-escaping and correct indentaion of the XML outout.

In \[18\]:

Copied!

```
print(qp_hybrid.to_xml())
```

print(qp_hybrid.to_xml())

```
<query-profile id="hybrid" type="hybrid-type">
  <field name="schema">doc</field>
  <field name="ranking.features.query(embedding)">embed(@query)</field>
  <field name="ranking.features.query(float_embedding)">embed(@query)</field>
  <field name="ranking.features.query(intercept)">
-7.798639
  </field>
  <field name="ranking.features.query(avg_top_3_chunk_sim_scores_param)">
13.38384
  </field>
  <field name="ranking.features.query(avg_top_3_chunk_text_scores_param)">
0.203145
  </field>
  <field name="ranking.features.query(bm25_chunks_param)">
0.159914
  </field>
  <field name="ranking.features.query(bm25_title_param)">
0.191867
  </field>
  <field name="ranking.features.query(max_chunk_sim_scores_param)">
10.067169
  </field>
  <field name="ranking.features.query(max_chunk_text_scores_param)">
0.153392
  </field>
  <field name="yql">select *
        from %{schema}
        where userInput(@query) or
        ({label:"title_label", targetHits:100}nearestNeighbor(title_embedding, embedding)) or
        ({label:"chunks_label", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))</field>
  <field name="hits">1</field>
  <field name="ranking.profile">learned-linear</field>
  <field name="presentation.summary">top_3_chunks</field>
</query-profile>
```

In \[19\]:

Copied!

```
print(qpt_hybrid.to_xml())
```

print(qpt_hybrid.to_xml())

```
<query-profile-type id="hybrid-type">
  <field name="ranking.features.query(embedding)" type="tensor&lt;int8&gt;(x[96])" mandatory="true" strict="true"></field>
  <field name="ranking.features.query(float_embedding)" type="tensor&lt;float&gt;(x[384])" mandatory="true" strict="true"></field>
</query-profile-type>
```

### Query profile variant[¶](#query-profile-variant)

See Vespa documentation on [Query Profile Variants](https://docs.vespa.ai/en/query-profiles.html#query-profile-variants) for more details.

In \[20\]:

Copied!

```
from vespa.configuration.query_profiles import *

qp_variant = query_profile(
    description("Multidimensional query profile"),
    dimensions("region,model,bucket"),
    field("My general a value", name="a"),
    query_profile(for_="us,nokia,test1")(
        field("My value of the combination us-nokia-test1-a", name="a"),
    ),
    query_profile(for_="us")(
        field("My value of the combination us-a", name="a"),
        field("My value of the combination us-b", name="b"),
    ),
    query_profile(for_="us,nokia,*")(
        field("My value of the combination us-nokia-a", name="a"),
        field("My value of the combination us-nokia-b", name="b"),
    ),
    query_profile(for_="us,*,test1")(
        field("My value of the combination us-test1-a", name="a"),
        field("My value of the combination us-test1-b", name="b"),
    ),
    id="multiprofile1",
)
```

from vespa.configuration.query_profiles import * qp_variant = query_profile( description("Multidimensional query profile"), dimensions("region,model,bucket"), field("My general a value", name="a"), query_profile(for\_="us,nokia,test1")( field("My value of the combination us-nokia-test1-a", name="a"), ), query_profile(for\_="us")( field("My value of the combination us-a", name="a"), field("My value of the combination us-b", name="b"), ), query_profile(for\_="us,nokia,\*")( field("My value of the combination us-nokia-a", name="a"), field("My value of the combination us-nokia-b", name="b"), ), query_profile(for\_="us,\*,test1")( field("My value of the combination us-test1-a", name="a"), field("My value of the combination us-test1-b", name="b"), ), id="multiprofile1", )

In \[21\]:

Copied!

```
from vespa.configuration.query_profiles import *

qpt_alias = query_profile_type(
    match_(path="true"),  # Match is sanitized due to python keyword
    field(
        name="ranking.features.query(query_embedding)",
        type="tensor<float>(x[512])",
        alias="q_emb query_emb",
    ),
    id="queryemb",
    inherits="native",
)
```

from vespa.configuration.query_profiles import * qpt_alias = query_profile_type( match\_(path="true"), # Match is sanitized due to python keyword field( name="ranking.features.query(query_embedding)", type="tensor<float>(x[512])", alias="q_emb query_emb", ), id="queryemb", inherits="native", )

You can pass this configuration to the `ApplicationPackage` when creating it, and it will be included in the generated `services.xml` file. Or, you can add it to the `ApplicationPackage` after it has been created by using the `add_query_profile` method:

In \[22\]:

Copied!

```
app_package.add_query_profile([qp_hybrid, qp_variant, qpt_hybrid, qpt_alias])
```

app_package.add_query_profile([qp_hybrid, qp_variant, qpt_hybrid, qpt_alias])

And by dumping the application package to files, we can see that all query profiles and query profile types are written to the `search/query-profiles` directory in the application package.

In \[23\]:

Copied!

```
import tempfile
import os

temp_dir = tempfile.mkdtemp()
app_package.to_files(temp_dir)
print(f"Application package files written to {temp_dir}")
print("Files in the temporary directory:")
print(os.listdir(temp_dir))
print("Files in the `search/query-profiles` directory:")
print(os.listdir(os.path.join(temp_dir, "search", "query-profiles")))
```

import tempfile import os temp_dir = tempfile.mkdtemp() app_package.to_files(temp_dir) print(f"Application package files written to {temp_dir}") print("Files in the temporary directory:") print(os.listdir(temp_dir)) print("Files in the `search/query-profiles` directory:") print(os.listdir(os.path.join(temp_dir, "search", "query-profiles")))

```
Application package files written to /var/folders/vb/ch14y_kn4mqfz75bhc9_g5980000gn/T/tmpyzrfju5a
Files in the temporary directory:
['services.xml', 'models', 'schemas', 'search', 'files']
Files in the `search/query-profiles` directory:
['types', 'multiprofile1.xml', 'hybrid.xml', 'default.xml']
```

Note that this combination of query profiles would not make sense to deploy together in the same application, but the point here is to demonstrate the flexibility of the new `query_profile_config` parameter, which should enable you to express any query profile or query profile type in python code, and add it to your `ApplicationPackage`.

The following xml-tags are available to construct query profiles and query profile types:

In \[24\]:

Copied!

```
queryprofile_tags
```

queryprofile_tags

Out\[24\]:

```
['query-profile',
 'query-profile-type',
 'field',
 'match',
 'strict',
 'description',
 'dimensions',
 'ref']
```

In order to avoid conflicts with Python reserved words, or commonly used objects, the following tags are (optionally) constructed by adding a `_` at the end of the tag name, or attribute name:

In \[25\]:

Copied!

```
from vespa.configuration.vt import restore_reserved

restore_reserved
```

from vespa.configuration.vt import restore_reserved restore_reserved

Out\[25\]:

```
{'type_': 'type',
 'class_': 'class',
 'for_': 'for',
 'time_': 'time',
 'io_': 'io',
 'from_': 'from',
 'match_': 'match'}
```

Note that we also here must sanitize the names of the `match` tag to avoid any conflicts with Python keyword, so `match` should be passed as `match_`. Additionally, we use the same approach as for the `ServicesConfiguration` object, so any hyphens in the tag names should be replaced with underscores.

## Configuring Deployment.xml[¶](#configuring-deploymentxml)

The `deployment.xml` configuration is used to specify how your Vespa application should be deployed across different environments and regions. This only applies to [Vespa Cloud](https://cloud.vespa.ai/) deployments, where you can specify deployment targets, regions, and deployment policies. For complete deployment configuration reference, see the [Vespa deployment.xml documentation](https://docs.vespa.ai/en/reference/deployment.html).

Similar to services.xml and query profiles, you can now express `deployment.xml` configuration using Python with the **Vespa Tag (VT)** syntax.

### Simple deployment configuration[¶](#simple-deployment-configuration)

Here's a basic example that deploys to two production regions:

In \[26\]:

Copied!

```
from vespa.configuration.deployment import deployment, prod, region
from vespa.package import ApplicationPackage

# Simple deployment to multiple regions
simple_deployment = deployment(
    prod(region("aws-us-east-1c"), region("aws-us-west-2a")), version="1.0"
)

app_package = ApplicationPackage(name="myapp", deployment_config=simple_deployment)
```

from vespa.configuration.deployment import deployment, prod, region from vespa.package import ApplicationPackage

# Simple deployment to multiple regions

simple_deployment = deployment( prod(region("aws-us-east-1c"), region("aws-us-west-2a")), version="1.0" ) app_package = ApplicationPackage(name="myapp", deployment_config=simple_deployment)

This configuration will generate a `deployment.xml` file that looks like this:

In \[28\]:

Copied!

```
print(app_package.deployment_config.to_xml())
```

print(app_package.deployment_config.to_xml())

```
<deployment version="1.0">
  <prod>
    <region>aws-us-east-1c</region>
    <region>aws-us-west-2a</region>
  </prod>
</deployment>
```

### Advanced deployment configuration[¶](#advanced-deployment-configuration)

For more complex scenarios, you can configure multiple instances, deployment delays, upgrade blocking windows, and endpoints:

In \[29\]:

Copied!

```
from vespa.configuration.deployment import (
    deployment,
    instance,
    prod,
    region,
    block_change,
    delay,
    parallel,
    steps,
    endpoints,
    endpoint,
)

# Complex deployment with multiple instances and advanced policies
complex_deployment = deployment(
    # Beta instance - simple deployment
    instance(prod(region("aws-us-east-1c")), id="beta"),
    # Default instance with advanced configuration
    instance(
        # Block changes during specific time windows
        block_change(
            revision="false", days="mon,wed-fri", hours="16-23", time_zone="UTC"
        ),
        prod(
            # First region
            region("aws-us-east-1c"),
            # Delay before next deployment
            delay(hours="3", minutes="7", seconds="13"),
            # Parallel deployment to multiple regions
            parallel(
                region("aws-us-west-1c"),
                # Sequential steps within parallel block
                steps(region("aws-eu-west-1a"), delay(hours="3")),
            ),
        ),
        # Configure endpoints for this instance
        endpoints(
            endpoint(region("aws-us-east-1c"), container_id="my-container-service")
        ),
        id="default",
    ),
    # Global endpoints across instances
    endpoints(
        endpoint(
            instance("beta", weight="1"),
            id="my-weighted-endpoint",
            container_id="my-container-service",
            region="aws-us-east-1c",
        )
    ),
    version="1.0",
)

app_package = ApplicationPackage(name="myapp", deployment_config=complex_deployment)
```

from vespa.configuration.deployment import ( deployment, instance, prod, region, block_change, delay, parallel, steps, endpoints, endpoint, )

# Complex deployment with multiple instances and advanced policies

complex_deployment = deployment(

# Beta instance - simple deployment

instance(prod(region("aws-us-east-1c")), id="beta"),

# Default instance with advanced configuration

instance(

# Block changes during specific time windows

block_change( revision="false", days="mon,wed-fri", hours="16-23", time_zone="UTC" ), prod(

# First region

region("aws-us-east-1c"),

# Delay before next deployment

delay(hours="3", minutes="7", seconds="13"),

# Parallel deployment to multiple regions

parallel( region("aws-us-west-1c"),

# Sequential steps within parallel block

steps(region("aws-eu-west-1a"), delay(hours="3")), ), ),

# Configure endpoints for this instance

endpoints( endpoint(region("aws-us-east-1c"), container_id="my-container-service") ), id="default", ),

# Global endpoints across instances

endpoints( endpoint( instance("beta", weight="1"), id="my-weighted-endpoint", container_id="my-container-service", region="aws-us-east-1c", ) ), version="1.0", ) app_package = ApplicationPackage(name="myapp", deployment_config=complex_deployment)

And the generated `deployment.xml` will include all specified configurations:

In \[30\]:

Copied!

```
print(app_package.deployment_config.to_xml())
```

print(app_package.deployment_config.to_xml())

```
<deployment version="1.0">
  <instance id="beta">
    <prod>
      <region>aws-us-east-1c</region>
    </prod>
  </instance>
  <instance id="default">
    <block-change revision="false" days="mon,wed-fri" hours="16-23" time-zone="UTC"></block-change>
    <prod>
      <region>aws-us-east-1c</region>
      <delay hours="3" minutes="7" seconds="13"></delay>
      <parallel>
        <region>aws-us-west-1c</region>
        <steps>
          <region>aws-eu-west-1a</region>
          <delay hours="3"></delay>
        </steps>
      </parallel>
    </prod>
    <endpoints>
      <endpoint container-id="my-container-service">
        <region>aws-us-east-1c</region>
      </endpoint>
    </endpoints>
  </instance>
  <endpoints>
    <endpoint id="my-weighted-endpoint" container-id="my-container-service" region="aws-us-east-1c">
      <instance weight="1">beta</instance>
    </endpoint>
  </endpoints>
</deployment>
```

```

```

This advanced configuration generates a comprehensive `deployment.xml` with:

- Multiple application instances (beta and default)
- Upgrade blocking windows to prevent deployments during peak hours
- Deployment delays and parallel deployment strategies
- Regional and cross-instance endpoint configurations

To see the available tags for each configuration category, you can print the corresponding tag lists:

In \[31\]:

Copied!

```
from vespa.configuration.deployment import deployment_tags
from vespa.configuration.query_profiles import queryprofile_tags
from vespa.configuration.services import services_tags

print(deployment_tags)
print(queryprofile_tags)
print(services_tags)
```

from vespa.configuration.deployment import deployment_tags from vespa.configuration.query_profiles import queryprofile_tags from vespa.configuration.services import services_tags print(deployment_tags) print(queryprofile_tags) print(services_tags)

```
['deployment', 'instance', 'prod', 'region', 'block-change', 'delay', 'parallel', 'steps', 'endpoints', 'endpoint', 'staging']
['query-profile', 'query-profile-type', 'field', 'match', 'strict', 'description', 'dimensions', 'ref']
['abortondocumenterror', 'accesslog', 'admin', 'adminserver', 'age', 'binding', 'bucket-splitting', 'cache', 'certificate', 'chain', 'chunk', 'client', 'clients', 'cluster-controller', 'clustercontroller', 'clustercontrollers', 'component', 'components', 'compression', 'concurrency', 'config', 'configserver', 'configservers', 'conservative', 'container', 'content', 'coverage', 'disk', 'disk-limit-factor', 'diskbloatfactor', 'dispatch', 'dispatch-policy', 'distribution', 'document', 'document-api', 'document-processing', 'document-token-id', 'documentprocessor', 'documents', 'engine', 'environment-variables', 'execution-mode', 'federation', 'feeding', 'filtering', 'flush-on-shutdown', 'flushstrategy', 'gpu', 'gpu-device', 'group', 'groups-allowed-down-ratio', 'handler', 'http', 'ignore-undefined-fields', 'include', 'index', 'init-progress-time', 'initialize', 'interop-threads', 'interval', 'intraop-threads', 'io', 'jvm', 'level', 'lidspace', 'logstore', 'maintenance', 'max-bloat-factor', 'max-concurrent', 'max-document-tokens', 'max-hits-per-partition', 'max-premature-crashes', 'max-query-tokens', 'max-tokens', 'max-wait-after-coverage-factor', 'maxage', 'maxfilesize', 'maxmemorygain', 'maxpendingbytes', 'maxpendingdocs', 'maxsize', 'maxsize-percent', 'mbusport', 'memory', 'memory-limit-factor', 'merges', 'min-active-docs-coverage', 'min-distributor-up-ratio', 'min-node-ratio-per-group', 'min-redundancy', 'min-storage-up-ratio', 'min-wait-after-coverage-factor', 'minimum', 'model', 'model-evaluation', 'models', 'native', 'niceness', 'node', 'nodes', 'onnx', 'onnx-execution-mode', 'onnx-gpu-device', 'onnx-interop-threads', 'onnx-intraop-threads', 'persearch', 'persistence-threads', 'pooling-strategy', 'prepend', 'processing', 'processor', 'proton', 'provider', 'prune', 'query', 'query-timeout', 'query-token-id', 'read', 'redundancy', 'removed-db', 'renderer', 'requestthreads', 'resource-limits', 'resources', 'retrydelay', 'retryenabled', 'route', 'search', 'searchable-copies', 'searcher', 'searchnode', 'secret-store', 'server', 'services', 'slobrok', 'slobroks', 'stable-state-period', 'store', 'summary', 'sync-transactionlog', 'term-score-threshold', 'threadpool', 'threads', 'time', 'timeout', 'token', 'tokenizer-model', 'top-k-probability', 'total', 'tracelevel', 'transactionlog', 'transformer-attention-mask', 'transformer-end-sequence-token', 'transformer-input-ids', 'transformer-mask-token', 'transformer-model', 'transformer-output', 'transformer-pad-token', 'transformer-start-sequence-token', 'transition-time', 'tuning', 'type', 'unpack', 'visibility-delay', 'visitors', 'warmup', 'zookeeper']
```

### No proper validation until deploy time[¶](#no-proper-validation-until-deploy-time)

Note that any attribute can be passed to the tag constructor, with no validation at construction time. You will still get validation at deploy time as usual though.

### Cleanup[¶](#cleanup)

In \[32\]:

Copied!

```
vespa_docker.container.stop()
vespa_docker.container.remove()
```

vespa_docker.container.stop() vespa_docker.container.remove()

## Next steps[¶](#next-steps)

This is just an intro into to the advanced configuration options available in Vespa. For more details, see the [Vespa documentation](https://docs.vespa.ai/en/reference/services.html).
