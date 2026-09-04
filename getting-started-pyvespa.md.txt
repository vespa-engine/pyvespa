# Hybrid Search - Quickstart[¶](#hybrid-search-quickstart)

This tutorial creates a hybrid text search application combining traditional keyword matching with semantic vector search (dense retrieval). It also demonstrates using [Vespa native embedder](https://docs.vespa.ai/en/embedding.html) functionality.

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

[Install pyvespa](https://pyvespa.readthedocs.io/) and start Docker Daemon, validate minimum 6G available:

In \[1\]:

Copied!

```
!pip3 install pyvespa
!docker info | grep "Total Memory"
```

!pip3 install pyvespa !docker info | grep "Total Memory"

## Create an application package[¶](#create-an-application-package)

The [application package](https://vespa-engine.github.io/pyvespa/api/vespa/package.md) has all the Vespa configuration files - create one from scratch:

In \[ \]:

Copied!

```
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Component,
    Parameter,
    FieldSet,
    GlobalPhaseRanking,
    Function,
)

package = ApplicationPackage(
    name="hybridsearch",
    schema=[
        Schema(
            name="doc",
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
                        bolding=True,
                    ),
                    Field(
                        name="embedding",
                        type="tensor<float>(x[384])",
                        indexing=[
                            'input title . " " . input body',
                            "embed",
                            "index",
                            "attribute",
                        ],
                        ann=HNSW(distance_metric="angular"),
                        is_document_field=False,
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["title", "body"])],
            rank_profiles=[
                RankProfile(
                    name="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    functions=[
                        Function(name="bm25sum", expression="bm25(title) + bm25(body)")
                    ],
                    first_phase="bm25sum",
                ),
                RankProfile(
                    name="semantic",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="closeness(field, embedding)",
                ),
                RankProfile(
                    name="fusion",
                    inherits="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="closeness(field, embedding)",
                    global_phase=GlobalPhaseRanking(
                        expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))",
                        rerank_count=1000,
                    ),
                ),
            ],
        )
    ],
    components=[
        Component(
            id="e5",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    "transformer-model",
                    {
                        "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/e5-small-v2-int8.onnx"
                    },
                ),
                Parameter(
                    "tokenizer-model",
                    {
                        "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/tokenizer.json"
                    },
                ),
            ],
        )
    ],
)
```

from vespa.package import ( ApplicationPackage, Field, Schema, Document, HNSW, RankProfile, Component, Parameter, FieldSet, GlobalPhaseRanking, Function, ) package = ApplicationPackage( name="hybridsearch", schema=\[ Schema( name="doc", document=Document( fields=\[ Field(name="id", type="string", indexing=["summary"]), Field( name="title", type="string", indexing=["index", "summary"], index="enable-bm25", ), Field( name="body", type="string", indexing=["index", "summary"], index="enable-bm25", bolding=True, ), Field( name="embedding", type="tensor<float>(x[384])", indexing=[ 'input title . " " . input body', "embed", "index", "attribute", ], ann=HNSW(distance_metric="angular"), is_document_field=False, ), \] ), fieldsets=\[FieldSet(name="default", fields=["title", "body"])\], rank_profiles=\[ RankProfile( name="bm25", inputs=\[("query(q)", "tensor<float>(x[384])")\], functions=[ Function(name="bm25sum", expression="bm25(title) + bm25(body)") ], first_phase="bm25sum", ), RankProfile( name="semantic", inputs=\[("query(q)", "tensor<float>(x[384])")\], first_phase="closeness(field, embedding)", ), RankProfile( name="fusion", inherits="bm25", inputs=\[("query(q)", "tensor<float>(x[384])")\], first_phase="closeness(field, embedding)", global_phase=GlobalPhaseRanking( expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))", rerank_count=1000, ), ), \], ) \], components=\[ Component( id="e5", type="hugging-face-embedder", parameters=[ Parameter( "transformer-model", { "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/e5-small-v2-int8.onnx" }, ), Parameter( "tokenizer-model", { "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/tokenizer.json" }, ), ], ) \], )

Note that the name cannot have `-` or `_`.

## Deploy the Vespa application[¶](#deploy-the-vespa-application)

Deploy `package` on the local machine using Docker, without leaving the notebook, by creating an instance of [VespaDocker](https://vespa-engine.github.io/pyvespa/api/vespa/deployment#vespa.deployment.VespaDocker). `VespaDocker` connects to the local Docker daemon socket and starts the [Vespa docker image](https://hub.docker.com/r/vespaengine/vespa/).

If this step fails, please check that the Docker daemon is running, and that the Docker daemon socket can be used by clients (Configurable under advanced settings in Docker Desktop).

In \[1\]:

Copied!

```
from vespa.deployment import VespaDocker

vespa_docker = VespaDocker()
app = vespa_docker.deploy(application_package=package)
```

from vespa.deployment import VespaDocker vespa_docker = VespaDocker() app = vespa_docker.deploy(application_package=package)

`app` now holds a reference to a [Vespa](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa) instance.

## Feeding documents to Vespa[¶](#feeding-documents-to-vespa)

In this example we use the [HF Datasets](https://huggingface.co/docs/datasets/index) library to stream the [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) dataset and index in our newly deployed Vespa instance. Read more about the [NFCorpus](https://huggingface.co/datasets/mteb/nfcorpus):

> NFCorpus is a full-text English retrieval data set for Medical Information Retrieval.

The following uses the [stream](https://huggingface.co/docs/datasets/stream) option of datasets to stream the data without downloading all the contents locally. The `map` functionality allows us to convert the dataset fields into the expected feed format for `pyvespa` which expects a dict with the keys `id` and `fields`:

`{ "id": "vespa-document-id", "fields": {"vespa_field": "vespa-field-value"}}`

In \[1\]:

Copied!

```
from datasets import load_dataset

dataset = load_dataset("BeIR/nfcorpus", "corpus", split="corpus", streaming=True)
vespa_feed = dataset.map(
    lambda x: {
        "id": x["_id"],
        "fields": {"title": x["title"], "body": x["text"], "id": x["_id"]},
    }
)
```

from datasets import load_dataset dataset = load_dataset("BeIR/nfcorpus", "corpus", split="corpus", streaming=True) vespa_feed = dataset.map( lambda x: { "id": x["\_id"], "fields": {"title": x["title"], "body": x["text"], "id": x["\_id"]}, } )

Now we can feed to Vespa using `feed_iterable` which accepts any `Iterable` and an optional callback function where we can check the outcome of each operation. The application is configured to use [embedding](https://docs.vespa.ai/en/embedding.html) functionality, that produce a vector embedding using a concatenation of the title and the body input fields. This step is computionally expensive. Read more about embedding inference in Vespa in the [Accelerating Transformer-based Embedding Retrieval with Vespa](https://blog.vespa.ai/accelerating-transformer-based-embedding-retrieval-with-vespa/).

In \[1\]:

Copied!

```
from vespa.io import VespaResponse, VespaQueryResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Error when feeding document {id}: {response.get_json()}")


app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)
```

from vespa.io import VespaResponse, VespaQueryResponse def callback(response: VespaResponse, id: str): if not response.is_successful(): print(f"Error when feeding document {id}: {response.get_json()}") app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)

## Querying Vespa[¶](#querying-vespa)

Using the [Vespa Query language](https://docs.vespa.ai/en/query-language.html) we can query the indexed data.

- Using a context manager `with app.syncio() as session` to handle connection pooling ([best practices](https://cloud.vespa.ai/en/http-best-practices))
- The query method accepts any valid Vespa [query api parameter](https://docs.vespa.ai/en/reference/query-api-reference.html) in `**kwargs`
- Vespa api parameter names that contains `.` must be sent as `dict` parameters in the `body` method argument

The following searches for `How Fruits and Vegetables Can Treat Asthma?` using different retrieval and [ranking](https://docs.vespa.ai/en/ranking.html) strategies.

In \[1\]:

Copied!

```
import pandas as pd


def display_hits_as_df(response: VespaQueryResponse, fields) -> pd.DataFrame:
    records = []
    for hit in response.hits:
        record = {}
        for field in fields:
            record[field] = hit["fields"][field]
        records.append(record)
    return pd.DataFrame(records)
```

import pandas as pd def display_hits_as_df(response: VespaQueryResponse, fields) -> pd.DataFrame: records = [] for hit in response.hits: record = {} for field in fields: record[field] = hit["fields"][field] records.append(record) return pd.DataFrame(records)

### Plain Keyword search[¶](#plain-keyword-search)

The following uses plain keyword search functionality with [bm25](https://docs.vespa.ai/en/reference/bm25.html) ranking, the `bm25` rank-profile was configured in the application package to use a linear combination of the bm25 score of the query terms against the title and the body field.

In \[1\]:

Copied!

```
with app.syncio(connections=1) as session:
    query = "How Fruits and Vegetables Can Treat Asthma?"
    response: VespaQueryResponse = session.query(
        yql="select * from sources * where userQuery() limit 5",
        query=query,
        ranking="bm25",
    )
    assert response.is_successful()
    print(display_hits_as_df(response, ["id", "title"]))
```

with app.syncio(connections=1) as session: query = "How Fruits and Vegetables Can Treat Asthma?" response: VespaQueryResponse = session.query( yql="select * from sources * where userQuery() limit 5", query=query, ranking="bm25", ) assert response.is_successful() print(display_hits_as_df(response, ["id", "title"]))

### Plain Semantic Search[¶](#plain-semantic-search)

The following uses dense vector representations of the query and the document and matching is performed and accelerated by Vespa's support for [approximate nearest neighbor search](https://docs.vespa.ai/en/approximate-nn-hnsw.html). The vector embedding representation of the text is obtained using Vespa's [embedder functionality](https://docs.vespa.ai/en/embedding.html#embedding-a-query-text).

In \[1\]:

Copied!

```
with app.syncio(connections=1) as session:
    query = "How Fruits and Vegetables Can Treat Asthma?"
    response: VespaQueryResponse = session.query(
        yql="select * from sources * where ({targetHits:1000}nearestNeighbor(embedding,q)) limit 5",
        query=query,
        ranking="semantic",
        body={"input.query(q)": f"embed({query})"},
    )
    assert response.is_successful()
    print(display_hits_as_df(response, ["id", "title"]))
```

with app.syncio(connections=1) as session: query = "How Fruits and Vegetables Can Treat Asthma?" response: VespaQueryResponse = session.query( yql="select * from sources * where ({targetHits:1000}nearestNeighbor(embedding,q)) limit 5", query=query, ranking="semantic", body={"input.query(q)": f"embed({query})"}, ) assert response.is_successful() print(display_hits_as_df(response, ["id", "title"]))

### Hybrid Search[¶](#hybrid-search)

This is one approach to combine the two retrieval strategies and where we use Vespa's support for [cross-hits feature normalization and reciprocal rank fusion](https://docs.vespa.ai/en/phased-ranking.html#cross-hit-normalization-including-reciprocal-rank-fusion). This functionality is exposed in the context of `global` re-ranking, after the distributed query retrieval execution which might span 1000s of nodes.

#### Hybrid search with the OR query operator[¶](#hybrid-search-with-the-or-query-operator)

This combines the two methods using logical disjunction (OR). Note that the first-phase expression in our `fusion` expression is only using the semantic score, this because usually semantic search provides better recall than sparse keyword search alone.

In \[1\]:

Copied!

```
with app.syncio(connections=1) as session:
    query = "How Fruits and Vegetables Can Treat Asthma?"
    response: VespaQueryResponse = session.query(
        yql="select * from sources * where userQuery() or ({targetHits:1000}nearestNeighbor(embedding,q)) limit 5",
        query=query,
        ranking="fusion",
        body={"input.query(q)": f"embed({query})"},
    )
    assert response.is_successful()
    print(display_hits_as_df(response, ["id", "title"]))
```

with app.syncio(connections=1) as session: query = "How Fruits and Vegetables Can Treat Asthma?" response: VespaQueryResponse = session.query( yql="select * from sources * where userQuery() or ({targetHits:1000}nearestNeighbor(embedding,q)) limit 5", query=query, ranking="fusion", body={"input.query(q)": f"embed({query})"}, ) assert response.is_successful() print(display_hits_as_df(response, ["id", "title"]))

#### Hybrid search with the RANK query operator[¶](#hybrid-search-with-the-rank-query-operator)

This combines the two methods using the [rank](https://docs.vespa.ai/en/reference/query-language-reference.html#rank) query operator. In this case we express that we want to retrieve the top-1000 documents using vector search, and then have sparse features like BM25 calculated as well (second operand of the rank operator). Finally the hits are re-ranked using the reciprocal rank fusion

In \[1\]:

Copied!

```
with app.syncio(connections=1) as session:
    query = "How Fruits and Vegetables Can Treat Asthma?"
    response: VespaQueryResponse = session.query(
        yql="select * from sources * where rank({targetHits:1000}nearestNeighbor(embedding,q), userQuery()) limit 5",
        query=query,
        ranking="fusion",
        body={"input.query(q)": f"embed({query})"},
    )
    assert response.is_successful()
    print(display_hits_as_df(response, ["id", "title"]))
```

with app.syncio(connections=1) as session: query = "How Fruits and Vegetables Can Treat Asthma?" response: VespaQueryResponse = session.query( yql="select * from sources * where rank({targetHits:1000}nearestNeighbor(embedding,q), userQuery()) limit 5", query=query, ranking="fusion", body={"input.query(q)": f"embed({query})"}, ) assert response.is_successful() print(display_hits_as_df(response, ["id", "title"]))

#### Hybrid search with filters[¶](#hybrid-search-with-filters)

In this example we add another query term to the yql, restricting the nearest neighbor search to only consider documents that have vegetable in the title.

In \[1\]:

Copied!

```
with app.syncio(connections=1) as session:
    query = "How Fruits and Vegetables Can Treat Asthma?"
    response: VespaQueryResponse = session.query(
        yql='select * from sources * where title contains "vegetable" and rank({targetHits:1000}nearestNeighbor(embedding,q), userQuery()) limit 5',
        query=query,
        ranking="fusion",
        body={"input.query(q)": f"embed({query})"},
    )
    assert response.is_successful()
    print(display_hits_as_df(response, ["id", "title"]))
```

with app.syncio(connections=1) as session: query = "How Fruits and Vegetables Can Treat Asthma?" response: VespaQueryResponse = session.query( yql='select * from sources * where title contains "vegetable" and rank({targetHits:1000}nearestNeighbor(embedding,q), userQuery()) limit 5', query=query, ranking="fusion", body={"input.query(q)": f"embed({query})"}, ) assert response.is_successful() print(display_hits_as_df(response, ["id", "title"]))

## Cleanup[¶](#cleanup)

In \[1\]:

Copied!

```
vespa_docker.container.stop()
vespa_docker.container.remove()
```

vespa_docker.container.stop() vespa_docker.container.remove()

## Next steps[¶](#next-steps)

This is just an intro into the capabilities of Vespa and pyvespa. Browse the site to learn more about schemas, feeding and queries - find more complex applications in [examples](https://vespa-engine.github.io/pyvespa/examples/index.md).
