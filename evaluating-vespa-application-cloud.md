# Evaluating a Vespa Application[¶](#evaluating-a-vespa-application)

We are often asked by users and customers what is the best retrieval and ranking strategy for a given use case. And even though we might sometimes have an intuition, we always recommend to set up experiments and do a proper quantitative evaluation.

> Models are temporary; Evals are forever.
>
> -Eugene Yan

Without a proper evaluation setup, you run the risk of settling for `lgtm@10` (looks good to me @ 10).

Then, if you deploy your application to users, you can be sure that you will get feedback of queries that does not produce relevant results. If you then try to optimize for that without knowing whether your tweaks are actually improving the overall quality of your search, you might end up with a system that is worse than the one you started with.

## So, what can you do?[¶](#so-what-can-you-do)

You can set up a proper evaluation pipeline, where you can test different ranking strategies, and see how they perform on a set of evaluation queries that act as a proxy for your real users' queries. This way, you can make informed decisions about what works best for your use case. If you collect real user interactions, it could be even better, but it is important to also keep the evaluation pipeline light enough so that you can run it both during development and in CI pipelines (possibly at different scales).

This guide will show how you easily can evaluate a Vespa application using pyvespa's `VespaMatchEvaluator` and `VespaEvaluator` class.

### Evaluate match-phase (retrieval) for recall[¶](#evaluate-match-phase-retrieval-for-recall)

The match-phase (or retrieval phase) in Vespa is perform to retrieve candidate documents to rank. Here, what we care about is that all possibly relevant documents are retrieved fast, without matching too many documents. If we match too many documents, latency will suffer, as all retrieved docs will be exposed to ranking. For an introduction to phased retrieval in Vespa, see the [docs](https://docs.vespa.ai/en/phased-ranking.html)

For this tutorial, we will evaluate and compare `weakAnd`, `nearestNeighbor`, as well as the combination of the two (using `OR`-operator).

### Evaluate ranking[¶](#evaluate-ranking)

We will define and compare 4 different ranking strategies in this guide:

1. `bm25` - Keyword-based retrieval and ranking - The solid baseline.
1. `semantic` - Vector search using cosine similarity (using <https://huggingface.co/intfloat/e5-small-v2> for embeddings)
1. `fusion`- Hybrid search (semantic+keyword). Combining BM25 and Semantic with [reciprocal rank fusion](https://docs.vespa.ai/en/phased-ranking.html#cross-hit-normalization-including-reciprocal-rank-fusion)
1. `atan_norm` - Hybrid search, combining BM25 and Semantic with [atan normalization](https://docs.vespa.ai/en/tutorials/hybrid-search.html#hybrid-ranking) as described in Aapo Tanskanen's [Guidebook to the State-of-the-Art Embeddings and Information Retrieval](https://www.linkedin.com/pulse/guidebook-state-of-the-art-embeddings-information-aapo-tanskanen-pc3mf/) (Originally proposed by [Seo et al. (2022)](https://www.mdpi.com/2227-7390/10/8/1335))

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

**Pre-requisite**: Create a tenant at [cloud.vespa.ai](https://cloud.vespa.ai/), save the tenant name.

## Install[¶](#install)

Install [pyvespa](https://pyvespa.readthedocs.io/) >= 0.53.0 and the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html). The Vespa CLI is used for data and control plane key management ([Vespa Cloud Security Guide](https://cloud.vespa.ai/en/security/guide)).

In \[ \]:

Copied!

```
!pip3 install pyvespa vespacli datasets pandas
```

!pip3 install pyvespa vespacli datasets pandas

## Configure application[¶](#configure-application)

In \[1\]:

Copied!

```
# Replace with your tenant name from the Vespa Cloud Console
tenant_name = "vespa-team"
# Replace with your application name (does not need to exist yet)
application = "evaluation"
schema_name = "doc"
```

# Replace with your tenant name from the Vespa Cloud Console

tenant_name = "vespa-team"

# Replace with your application name (does not need to exist yet)

application = "evaluation" schema_name = "doc"

## Create an application package[¶](#create-an-application-package)

The [application package](https://vespa-engine.github.io/pyvespa/api/vespa/package.md) has all the Vespa configuration files - create one from scratch:

In \[2\]:

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
import pandas as pd


package = ApplicationPackage(
    name=application,
    schema=[
        Schema(
            name=schema_name,
            document=Document(
                fields=[
                    # Note that we need an id field as attribute to be able to do evaluation
                    # Vespa internal query document id is used as fallback, but have some limitations, see https://docs.vespa.ai/en/document-v1-api-guide.html#query-result-id
                    Field(name="id", type="string", indexing=["summary", "attribute"]),
                    Field(
                        name="text",
                        type="string",
                        indexing=["index", "summary"],
                        index="enable-bm25",
                        bolding=True,
                    ),
                    Field(
                        name="embedding",
                        type="tensor<float>(x[384])",
                        indexing=[
                            "input text",
                            "embed",  # uses default model
                            "index",
                            "attribute",
                        ],
                        ann=HNSW(distance_metric="angular"),
                        is_document_field=False,
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["text"])],
            rank_profiles=[
                RankProfile(
                    name="match-only",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="random",  # TODO: Remove when pyvespa supports empty first_phase
                ),
                RankProfile(
                    name="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    functions=[Function(name="bm25text", expression="bm25(text)")],
                    first_phase="bm25text",
                    match_features=["bm25text"],
                ),
                RankProfile(
                    name="semantic",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    functions=[
                        Function(
                            name="cos_sim", expression="closeness(field, embedding)"
                        )
                    ],
                    first_phase="cos_sim",
                    match_features=["cos_sim"],
                ),
                RankProfile(
                    name="fusion",
                    inherits="bm25",
                    functions=[
                        Function(
                            name="cos_sim", expression="closeness(field, embedding)"
                        )
                    ],
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="cos_sim",
                    global_phase=GlobalPhaseRanking(
                        expression="reciprocal_rank_fusion(bm25text, closeness(field, embedding))",
                        rerank_count=1000,
                    ),
                    match_features=["cos_sim", "bm25text"],
                ),
                RankProfile(
                    name="atan_norm",
                    inherits="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    functions=[
                        Function(
                            name="scale",
                            args=["val"],
                            expression="2*atan(val)/(3.14159)",
                        ),
                        Function(
                            name="normalized_bm25", expression="scale(bm25(text))"
                        ),
                        Function(
                            name="cos_sim", expression="closeness(field, embedding)"
                        ),
                    ],
                    first_phase="normalized_bm25",
                    global_phase=GlobalPhaseRanking(
                        expression="normalize_linear(normalized_bm25) + normalize_linear(cos_sim)",
                        rerank_count=1000,
                    ),
                    match_features=["cos_sim", "normalized_bm25"],
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
                        "model-id": "e5-small-v2"
                    },  # in vespa cloud, we can use the model-id for selected models, see https://cloud.vespa.ai/en/model-hub
                ),
                Parameter(
                    "tokenizer-model",
                    {"model-id": "e5-base-v2-vocab"},
                ),
            ],
        )
    ],
)
```

from vespa.package import ( ApplicationPackage, Field, Schema, Document, HNSW, RankProfile, Component, Parameter, FieldSet, GlobalPhaseRanking, Function, ) import pandas as pd package = ApplicationPackage( name=application, schema=\[ Schema( name=schema_name, document=Document( fields=\[

# Note that we need an id field as attribute to be able to do evaluation

# Vespa internal query document id is used as fallback, but have some limitations, see https://docs.vespa.ai/en/document-v1-api-guide.html#query-result-id

Field(name="id", type="string", indexing=["summary", "attribute"]), Field( name="text", type="string", indexing=["index", "summary"], index="enable-bm25", bolding=True, ), Field( name="embedding", type="tensor<float>(x[384])", indexing=[ "input text", "embed", # uses default model "index", "attribute", ], ann=HNSW(distance_metric="angular"), is_document_field=False, ), \] ), fieldsets=\[FieldSet(name="default", fields=["text"])\], rank_profiles=\[ RankProfile( name="match-only", inputs=\[("query(q)", "tensor<float>(x[384])")\], first_phase="random", # TODO: Remove when pyvespa supports empty first_phase ), RankProfile( name="bm25", inputs=\[("query(q)", "tensor<float>(x[384])")\], functions=[Function(name="bm25text", expression="bm25(text)")], first_phase="bm25text", match_features=["bm25text"], ), RankProfile( name="semantic", inputs=\[("query(q)", "tensor<float>(x[384])")\], functions=[ Function( name="cos_sim", expression="closeness(field, embedding)" ) ], first_phase="cos_sim", match_features=["cos_sim"], ), RankProfile( name="fusion", inherits="bm25", functions=[ Function( name="cos_sim", expression="closeness(field, embedding)" ) ], inputs=\[("query(q)", "tensor<float>(x[384])")\], first_phase="cos_sim", global_phase=GlobalPhaseRanking( expression="reciprocal_rank_fusion(bm25text, closeness(field, embedding))", rerank_count=1000, ), match_features=["cos_sim", "bm25text"], ), RankProfile( name="atan_norm", inherits="bm25", inputs=\[("query(q)", "tensor<float>(x[384])")\], functions=\[ Function( name="scale", args=["val"], expression="2\*atan(val)/(3.14159)", ), Function( name="normalized_bm25", expression="scale(bm25(text))" ), Function( name="cos_sim", expression="closeness(field, embedding)" ), \], first_phase="normalized_bm25", global_phase=GlobalPhaseRanking( expression="normalize_linear(normalized_bm25) + normalize_linear(cos_sim)", rerank_count=1000, ), match_features=["cos_sim", "normalized_bm25"], ), \], ) \], components=\[ Component( id="e5", type="hugging-face-embedder", parameters=[ Parameter( "transformer-model", { "model-id": "e5-small-v2" }, # in vespa cloud, we can use the model-id for selected models, see https://cloud.vespa.ai/en/model-hub ), Parameter( "tokenizer-model", {"model-id": "e5-base-v2-vocab"}, ), ], ) \], )

Note that the name cannot have `-` or `_`.

## Deploy to Vespa Cloud[¶](#deploy-to-vespa-cloud)

The app is now defined and ready to deploy to Vespa Cloud.

Deploy `package` to Vespa Cloud, by creating an instance of [VespaCloud](https://vespa-engine.github.io/pyvespa/api/vespa/deployment#VespaCloud):

In \[3\]:

Copied!

```
from vespa.deployment import VespaCloud
import os

# Key is only used for CI/CD. Can be removed if logging in interactively

vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=application,
    key_content=os.getenv(
        "VESPA_TEAM_API_KEY", None
    ),  # Key is only used for CI/CD. Can be removed if logging in interactively
    application_package=package,
)
```

from vespa.deployment import VespaCloud import os

# Key is only used for CI/CD. Can be removed if logging in interactively

vespa_cloud = VespaCloud( tenant=tenant_name, application=application, key_content=os.getenv( "VESPA_TEAM_API_KEY", None ), # Key is only used for CI/CD. Can be removed if logging in interactively application_package=package, )

```
Setting application...
Running: vespa config set application vespa-team.evaluation.default
Setting target cloud...
Running: vespa config set target cloud

Api-key found for control plane access. Using api-key.
```

For more details on different authentication options and methods, see [authenticating-to-vespa-cloud](https://vespa-engine.github.io/pyvespa/authenticating-to-vespa-cloud.md).

The following will upload the application package to Vespa Cloud Dev Zone (`aws-us-east-1c`), read more about [Vespa Zones](https://cloud.vespa.ai/en/reference/zones.html). The Vespa Cloud Dev Zone is considered as a sandbox environment where resources are down-scaled and idle deployments are expired automatically. For information about production deployments, see the following [method](https://vespa-engine.github.io/pyvespa/api/vespa/deployment#vespa.deployment.VespaCloud.deploy_to_prod).

> Note: Deployments to dev and perf expire after 7 days of inactivity, i.e., 7 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 7 more days.

Now deploy the app to Vespa Cloud dev zone.

The first deployment typically takes 2 minutes until the endpoint is up. (Applications that for example refer to large onnx-models may take a bit longer.)

In \[4\]:

Copied!

```
from vespa.application import Vespa

app: Vespa = vespa_cloud.deploy()
```

from vespa.application import Vespa app: Vespa = vespa_cloud.deploy()

```
Deployment started in run 52 of dev-aws-us-east-1c for vespa-team.evaluation. This may take a few minutes the first time.
INFO    [06:52:41]  Deploying platform version 8.586.25 and application dev build 50 for dev-aws-us-east-1c of default ...
INFO    [06:52:41]  Using CA signed certificate version 9
INFO    [06:52:42]  Using 1 nodes in container cluster 'evaluation_container'
INFO    [06:52:44]  Session 379645 for tenant 'vespa-team' prepared and activated.
INFO    [06:52:44]  ######## Details for all nodes ########
INFO    [06:52:45]  h125699b.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:52:45]  --- platform vespa/cloud-tenant-rhel8:8.586.25
INFO    [06:52:45]  --- storagenode on port 19102 has config generation 379643, wanted is 379645
INFO    [06:52:45]  --- searchnode on port 19107 has config generation 379643, wanted is 379645
INFO    [06:52:45]  --- distributor on port 19111 has config generation 379645, wanted is 379645
INFO    [06:52:45]  --- metricsproxy-container on port 19092 has config generation 379645, wanted is 379645
INFO    [06:52:45]  h119183e.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:52:45]  --- platform vespa/cloud-tenant-rhel8:8.586.25
INFO    [06:52:45]  --- container-clustercontroller on port 19050 has config generation 379643, wanted is 379645
INFO    [06:52:45]  --- metricsproxy-container on port 19092 has config generation 379645, wanted is 379645
INFO    [06:52:45]  h125689a.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:52:45]  --- platform vespa/cloud-tenant-rhel8:8.586.25
INFO    [06:52:45]  --- container on port 4080 has config generation 379643, wanted is 379645
INFO    [06:52:45]  --- metricsproxy-container on port 19092 has config generation 379643, wanted is 379645
INFO    [06:52:45]  h97530b.dev.us-east-1c.aws.vespa-cloud.net: expected to be UP
INFO    [06:52:45]  --- platform vespa/cloud-tenant-rhel8:8.586.25
INFO    [06:52:45]  --- logserver-container on port 4080 has config generation 379643, wanted is 379645
INFO    [06:52:45]  --- metricsproxy-container on port 19092 has config generation 379643, wanted is 379645
INFO    [06:52:56]  Found endpoints:
INFO    [06:52:56]  - dev.aws-us-east-1c
INFO    [06:52:56]   |-- https://f4f49447.ccc9bd09.z.vespa-app.cloud/ (cluster 'evaluation_container')
INFO    [06:52:56]  Deployment of new application revision complete!
Only region: aws-us-east-1c available in dev environment.
Found mtls endpoint for evaluation_container
URL: https://f4f49447.ccc9bd09.z.vespa-app.cloud/
Application is up!
```

If the deployment failed, it is possible you forgot to add the key in the Vespa Cloud Console in the `vespa auth api-key` step above.

If you can authenticate, you should see lines like the following

```
 Deployment started in run 1 of dev-aws-us-east-1c for mytenant.hybridsearch.
```

The deployment takes a few minutes the first time while Vespa Cloud sets up the resources for your Vespa application

`app` now holds a reference to a [Vespa](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa) instance. We can access the mTLS protected endpoint name using the control-plane (vespa_cloud) instance. This endpoint we can query and feed to (data plane access) using the mTLS certificate generated in previous steps.

See [Authenticating to Vespa Cloud](https://vespa-engine.github.io/pyvespa/authenticating-to-vespa-cloud.md) for details on using token authentication instead of mTLS.

## Getting your evaluation data[¶](#getting-your-evaluation-data)

For evaluating information retrieval methods, in addition to the document corpus, we also need a set of queries and a mapping from queries to relevant documents.

For this guide, we will use the [NanoMSMARCO](https://huggingface.co/datasets/zeta-alpha-ai/NanoMSMARCO) dataset, made available on huggingface by [Zeta Alpha](https://zeta-alpha.com/).

This dataset is a subset of their 🍺[NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6)-collection, with 50 queries and up to 10K documents each.

This is a great dataset for testing and evaluating information retrieval methods quickly, as it is small and easy to work with.

Note that for almost any real-world use case, we would recommend you to create your own evaluation dataset. See [Vespa blog post](https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/) on how you can get help from an LLM for this.

Note that creating 20-50 queries and annotating relevant documents for each query can be a good start and well worth the effort.

In \[5\]:

Copied!

```
from datasets import load_dataset

dataset_id = "zeta-alpha-ai/NanoMSMARCO"

dataset = load_dataset(dataset_id, "corpus", split="train", streaming=True)
vespa_feed = dataset.map(
    lambda x: {
        "id": x["_id"],
        "fields": {"text": x["text"], "id": x["_id"]},
    }
)
```

from datasets import load_dataset dataset_id = "zeta-alpha-ai/NanoMSMARCO" dataset = load_dataset(dataset_id, "corpus", split="train", streaming=True) vespa_feed = dataset.map( lambda x: { "id": x["\_id"], "fields": {"text": x["text"], "id": x["\_id"]}, } )

Note that we are only *evaluating* rank strategies here, we consider it OK to use the `train` split for evaluation. If we were to make changes to our ranking strategies, such as adding weighting terms, or training ml models for ranking, we would suggest to adopt a `train`, `validation`, `test` split approach to avoid overfitting.

In \[6\]:

Copied!

```
query_ds = load_dataset(dataset_id, "queries", split="train")
qrels = load_dataset(dataset_id, "qrels", split="train")
```

query_ds = load_dataset(dataset_id, "queries", split="train") qrels = load_dataset(dataset_id, "qrels", split="train")

In \[7\]:

Copied!

```
ids_to_query = dict(zip(query_ds["_id"], query_ds["text"]))
```

ids_to_query = dict(zip(query_ds["\_id"], query_ds["text"]))

Let us print the first 5 queries:

In \[8\]:

Copied!

```
for idx, (qid, q) in enumerate(ids_to_query.items()):
    print(f"qid: {qid}, query: {q}")
    if idx == 5:
        break
```

for idx, (qid, q) in enumerate(ids_to_query.items()): print(f"qid: {qid}, query: {q}") if idx == 5: break

```
qid: 994479, query: which health care system provides all citizens or residents with equal access to health care services
qid: 1009388, query: what's right in health care
qid: 1088332, query: weather in oran
qid: 265729, query: how long keep financial records
qid: 1099433, query: how do hoa fees work
qid: 200600, query: heels or heal
```

In \[9\]:

Copied!

```
relevant_docs = dict(zip(qrels["query-id"], qrels["corpus-id"]))
```

relevant_docs = dict(zip(qrels["query-id"], qrels["corpus-id"]))

Let us print the first 5 query ids and their relevant documents:

In \[10\]:

Copied!

```
for idx, (qid, doc_id) in enumerate(relevant_docs.items()):
    print(f"qid: {qid}, doc_id: {doc_id}")
    if idx == 5:
        break
```

for idx, (qid, doc_id) in enumerate(relevant_docs.items()): print(f"qid: {qid}, doc_id: {doc_id}") if idx == 5: break

```
qid: 994479, doc_id: 7275120
qid: 1009388, doc_id: 7248824
qid: 1088332, doc_id: 7094398
qid: 265729, doc_id: 7369987
qid: 1099433, doc_id: 7255675
qid: 200600, doc_id: 7929603
```

We can see that this dataset only has one relevant document per query. The `VespaEvaluator` class handles this just fine, but you could also provide a set of relevant documents per query if there are multiple relevant docs.

```
# multiple relevant docs per query
qrels = {
    "q1": {"doc1", "doc2"},
    "q2": {"doc3", "doc4"},
    # etc. 
}
```

Now we can feed to Vespa using `feed_iterable` which accepts any `Iterable` and an optional callback function where we can check the outcome of each operation. The application is configured to use [embedding](https://docs.vespa.ai/en/embedding.html) functionality, that produce a vector embedding using a concatenation of the title and the body input fields. This step may be resource intensive, depending on the model size.

Read more about embedding inference in Vespa in the [Accelerating Transformer-based Embedding Retrieval with Vespa](https://blog.vespa.ai/accelerating-transformer-based-embedding-retrieval-with-vespa/) blog post.

Default node resources in Vespa Cloud have 2 v-cpu for the Dev Zone.

In \[11\]:

Copied!

```
from vespa.io import VespaResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Error when feeding document {id}: {response.get_json()}")


app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)
```

from vespa.io import VespaResponse def callback(response: VespaResponse, id: str): if not response.is_successful(): print(f"Error when feeding document {id}: {response.get_json()}") app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)

## Evaluate match-phase[¶](#evaluate-match-phase)

There are two separate classes provided for doing evaluations:

1. `VespaMatchEvaluator`, which is intended to evaluate only the *retrieval* (or match-phase), and should not do any ranking. This is useful to evaluate whether your relevant documents are retrieved (and thus exposed to ranking). It only computes recall, total matched documents per query as well as `searchtime`.
1. `VespaEvaluator` is intended to evaluate a complete ranking strategy, across several common IR metrics.

Both API's are inspired by [SentenceTransformers](https://www.sbert.net/) [`InformationRetrievalEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#informationretrievalevaluator) class.

The difference it that `VespaMatchEvaluator` evaluates only the retrieval phase, while `VespaEvaluator` evaluates your whole retrieval and ranking *system* (Vespa application) as opposed to a single model. Your application should be fed with the document corpus in advance, instead of taking in the document corpus.

We now have created the app, the queries, and the relevant documents. The only thing missing before we can initialize the `VespaMatchEvaluator` is a set of functions that defines the Vespa queries. Each of them is passed as `vespa_query_fn`.

We will use the `vespa.querybuilder` module to create the queries. See [reference doc](https://vespa-engine.github.io/pyvespa/api/vespa/querybuilder/builder/builder.md) and [example notebook](https://vespa-engine.github.io/pyvespa/query.md#Using-the-Querybuilder-DSL-API) for more details on usage.

This module is a Python wrapper around the Vespa Query Language (YQL), which is an alternative to providing the YQL query as a string directly.

In \[12\]:

Copied!

```
import vespa.querybuilder as qb


def match_weakand_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": str(qb.select("*").from_(schema_name).where(qb.userQuery(query_text))),
        "query": query_text,
        "ranking": "match-only",
        "input.query(q)": f"embed({query_text})",
    }


def match_hybrid_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": str(
            qb.select("*")
            .from_(schema_name)
            .where(
                qb.nearestNeighbor(
                    field="embedding",
                    query_vector="q",
                    annotations={"targetHits": 100},
                )
                | qb.userQuery(
                    query_text,
                )
            )
        ),
        "query": query_text,
        "ranking": "match-only",
        "input.query(q)": f"embed({query_text})",
    }


def match_semantic_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": str(
            qb.select("*")
            .from_(schema_name)
            .where(
                qb.nearestNeighbor(
                    field="embedding",
                    query_vector="q",
                    annotations={"targetHits": 100},
                )
            )
        ),
        "query": query_text,
        "ranking": "match-only",
        "input.query(q)": f"embed({query_text})",
    }
```

import vespa.querybuilder as qb def match_weakand_query_fn(query_text: str, top_k: int) -> dict: return { "yql": str(qb.select("\*").from\_(schema_name).where(qb.userQuery(query_text))), "query": query_text, "ranking": "match-only", "input.query(q)": f"embed({query_text})", } def match_hybrid_query_fn(query_text: str, top_k: int) -> dict: return { "yql": str( qb.select("\*") .from\_(schema_name) .where( qb.nearestNeighbor( field="embedding", query_vector="q", annotations={"targetHits": 100}, ) | qb.userQuery( query_text, ) ) ), "query": query_text, "ranking": "match-only", "input.query(q)": f"embed({query_text})", } def match_semantic_query_fn(query_text: str, top_k: int) -> dict: return { "yql": str( qb.select("\*") .from\_(schema_name) .where( qb.nearestNeighbor( field="embedding", query_vector="q", annotations={"targetHits": 100}, ) ) ), "query": query_text, "ranking": "match-only", "input.query(q)": f"embed({query_text})", }

Now, let us run the evaluator:

In \[13\]:

Copied!

```
from vespa.evaluation import VespaMatchEvaluator


match_results = {}
for evaluator_name, query_fn in [
    ("semantic", match_semantic_query_fn),
    ("weakand", match_weakand_query_fn),
    ("hybrid", match_hybrid_query_fn),
]:
    print(f"Evaluating {evaluator_name}...")

    match_evaluator = VespaMatchEvaluator(
        queries=ids_to_query,
        relevant_docs=relevant_docs,
        vespa_query_fn=query_fn,
        app=app,
        name="test-run",
        id_field="id",  # specify the id field used in the relevant_docs
        write_csv=True,
        write_verbose=True,  # optionally write verbose metrics to CSV
    )

    results = match_evaluator()
    match_results[evaluator_name] = results
    print(f"Results for {evaluator_name}:")
    print(results)
```

from vespa.evaluation import VespaMatchEvaluator match_results = {} for evaluator_name, query_fn in \[ ("semantic", match_semantic_query_fn), ("weakand", match_weakand_query_fn), ("hybrid", match_hybrid_query_fn), \]: print(f"Evaluating {evaluator_name}...") match_evaluator = VespaMatchEvaluator( queries=ids_to_query, relevant_docs=relevant_docs, vespa_query_fn=query_fn, app=app, name="test-run", id_field="id", # specify the id field used in the relevant_docs write_csv=True, write_verbose=True, # optionally write verbose metrics to CSV ) results = match_evaluator() match_results[evaluator_name] = results print(f"Results for {evaluator_name}:") print(results)

```
Evaluating semantic...
Results for semantic:
{'match_recall': 1.0, 'avg_recall_per_query': 1.0, 'total_relevant_docs': 50, 'total_matched_relevant': 50, 'avg_matched_per_query': 100.0, 'total_queries': 50, 'searchtime_avg': 0.0535, 'searchtime_q50': 0.053, 'searchtime_q90': 0.0786, 'searchtime_q95': 0.08700000000000001}
Evaluating weakand...
Results for weakand:
{'match_recall': 0.98, 'avg_recall_per_query': 0.98, 'total_relevant_docs': 50, 'total_matched_relevant': 49, 'avg_matched_per_query': 809.86, 'total_queries': 50, 'searchtime_avg': 0.04391999999999998, 'searchtime_q50': 0.043000000000000003, 'searchtime_q90': 0.058300000000000005, 'searchtime_q95': 0.06665}
Evaluating hybrid...
Results for hybrid:
{'match_recall': 1.0, 'avg_recall_per_query': 1.0, 'total_relevant_docs': 50, 'total_matched_relevant': 50, 'avg_matched_per_query': 833.18, 'total_queries': 50, 'searchtime_avg': 0.03699999999999999, 'searchtime_q50': 0.037, 'searchtime_q90': 0.0531, 'searchtime_q95': 0.058299999999999984}
```

By setting `write_csv=True` and `verbose=True`, we can save a CSV-file for each query to inspect further the queries that were not matched. This is important to understand how you could improve recall if some relevant documents were not matched.

In \[14\]:

Copied!

```
results = pd.DataFrame(match_results)
results
```

results = pd.DataFrame(match_results) results

Out\[14\]:

|                        | semantic | weakand   | hybrid   |
| ---------------------- | -------- | --------- | -------- |
| match_recall           | 1.0000   | 0.98000   | 1.0000   |
| avg_recall_per_query   | 1.0000   | 0.98000   | 1.0000   |
| total_relevant_docs    | 50.0000  | 50.00000  | 50.0000  |
| total_matched_relevant | 50.0000  | 49.00000  | 50.0000  |
| avg_matched_per_query  | 100.0000 | 809.86000 | 833.1800 |
| total_queries          | 50.0000  | 50.00000  | 50.0000  |
| searchtime_avg         | 0.0535   | 0.04392   | 0.0370   |
| searchtime_q50         | 0.0530   | 0.04300   | 0.0370   |
| searchtime_q90         | 0.0786   | 0.05830   | 0.0531   |
| searchtime_q95         | 0.0870   | 0.06665   | 0.0583   |

Here, we can see that all retrieval strategies actually match all the relevant documents. To tune number of documents retrieved and latency, we could tune the `targetHits`-parameter for both the `nearestNeighbor`-operator and `weakAnd`-parameter (our `userInput` is converted to `weakAnd`, see [docs](https://docs.vespa.ai/en/reference/query-language-reference.html)), as well as several additional `weakAnd`-parameters. See Vespa [blog](https://blog.vespa.ai/tripling-the-query-performance-of-lexical-search/) for details. We will not go in detail of this in this notebook.

## Evaluate ranking[¶](#evaluate-ranking)

Now, we will move on to demonstrate how to evaluate the ranking strategies. For that, we will use the `VespaEvaluator`-class. Its interface is very similar to `VespaMatchEvaluator`, with the difference that it has much more metrics available. Also note that number of `hits` will affect the number of documents considered for evaluation.

## VespaEvaluator[¶](#vespaevaluator)

Let us take a look at its API and documentation:

In \[15\]:

Copied!

```
from vespa.evaluation import VespaEvaluator

?VespaEvaluator
```

from vespa.evaluation import VespaEvaluator ?VespaEvaluator

````
Init signature:
VespaEvaluator(
    queries: 'Dict[str, str]',
    relevant_docs: 'Union[Dict[str, Union[Set[str], Dict[str, float]]], Dict[str, str]]',
    vespa_query_fn: 'Callable[[str, int, Optional[str]], dict]',
    app: 'Vespa',
    name: 'str' = '',
    id_field: 'str' = '',
    accuracy_at_k: 'List[int]' = [1, 3, 5, 10],
    precision_recall_at_k: 'List[int]' = [1, 3, 5, 10],
    mrr_at_k: 'List[int]' = [10],
    ndcg_at_k: 'List[int]' = [10],
    map_at_k: 'List[int]' = [100],
    write_csv: 'bool' = False,
    csv_dir: 'Optional[str]' = None,
)
Docstring:     
Evaluate retrieval performance on a Vespa application.

This class:

- Iterates over queries and issues them against your Vespa application.
- Retrieves top-k documents per query (with k = max of your IR metrics).
- Compares the retrieved documents with a set of relevant document ids.
- Computes IR metrics: Accuracy@k, Precision@k, Recall@k, MRR@k, NDCG@k, MAP@k.
- Logs vespa search times for each query.
- Logs/returns these metrics.
- Optionally writes out to CSV.

Note: The 'id_field' needs to be marked as an attribute in your Vespa schema, so filtering can be done on it.


Example usage:
    ```python
    from vespa.application import Vespa
    from vespa.evaluation import VespaEvaluator

    queries = {
        "q1": "What is the best GPU for gaming?",
        "q2": "How to bake sourdough bread?",
        # ...
    }
    relevant_docs = {
        "q1": {"d12", "d99"},
        "q2": {"d101"},
        # ...
    }
    # relevant_docs can also be a dict of query_id => single relevant doc_id
    # relevant_docs = {
    #     "q1": "d12",
    #     "q2": "d101",
    #     # ...
    # }
    # Or, relevant_docs can be a dict of query_id => map of doc_id => relevance
    # relevant_docs = {
    #     "q1": {"d12": 1, "d99": 0.1},
    #     "q2": {"d101": 0.01},
    #     # ...
    # Note that for non-binary relevance, the relevance values should be in [0, 1], and that
    # only the nDCG metric will be computed.

    def my_vespa_query_fn(query_text: str, top_k: int) -> dict:
        return {
            "yql": 'select * from sources * where userInput("' + query_text + '");',
            "hits": top_k,
            "ranking": "your_ranking_profile",
        }

    app = Vespa(url="http://localhost", port=8080)

    evaluator = VespaEvaluator(
        queries=queries,
        relevant_docs=relevant_docs,
        vespa_query_fn=my_vespa_query_fn,
        app=app,
        name="test-run",
        accuracy_at_k=[1, 3, 5],
        precision_recall_at_k=[1, 3, 5],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[100],
        write_csv=True
    )

    results = evaluator()
    print("Primary metric:", evaluator.primary_metric)
    print("All results:", results)
    ```

Args:
    queries (Dict[str, str]): A dictionary where keys are query IDs and values are query strings.
    relevant_docs (Union[Dict[str, Union[Set[str], Dict[str, float]]], Dict[str, str]]):
        A dictionary mapping query IDs to their relevant document IDs.
        Can be a set of doc IDs for binary relevance, a dict of doc_id to relevance score (float between 0 and 1)
        for graded relevance, or a single doc_id string.
    vespa_query_fn (Callable[[str, int, Optional[str]], dict]): A function that takes a query string,
        the number of hits to retrieve (top_k), and an optional query_id, and returns a Vespa query body dictionary.
    app (Vespa): An instance of the Vespa application.
    name (str, optional): A name for this evaluation run. Defaults to "".
    id_field (str, optional): The field name in the Vespa hit that contains the document ID.
        If empty, it tries to infer the ID from the 'id' field or 'fields.id'. Defaults to "".
    accuracy_at_k (List[int], optional): List of k values for which to compute Accuracy@k.
        Defaults to [1, 3, 5, 10].
    precision_recall_at_k (List[int], optional): List of k values for which to compute Precision@k and Recall@k.
        Defaults to [1, 3, 5, 10].
    mrr_at_k (List[int], optional): List of k values for which to compute MRR@k. Defaults to [10].
    ndcg_at_k (List[int], optional): List of k values for which to compute NDCG@k. Defaults to [10].
    map_at_k (List[int], optional): List of k values for which to compute MAP@k. Defaults to [100].
    write_csv (bool, optional): Whether to write the evaluation results to a CSV file. Defaults to False.
    csv_dir (Optional[str], optional): Directory to save the CSV file. Defaults to None (current directory).
File:           ~/Repos/pyvespa/vespa/evaluation.py
Type:           ABCMeta
Subclasses:
````

In \[16\]:

Copied!

```
def semantic_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": str(
            qb.select("*")
            .from_(schema_name)
            .where(
                qb.nearestNeighbor(
                    field="embedding",
                    query_vector="q",
                    annotations={"targetHits": 100},
                )
            )
        ),
        "query": query_text,
        "ranking": "semantic",
        "input.query(q)": f"embed({query_text})",
        "hits": top_k,
    }


def bm25_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": "select * from sources * where userQuery();",  # provide the yql directly as a string
        "query": query_text,
        "ranking": "bm25",
        "hits": top_k,
    }


def fusion_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": str(
            qb.select("*")
            .from_(schema_name)
            .where(
                qb.nearestNeighbor(
                    field="embedding",
                    query_vector="q",
                    annotations={"targetHits": 100},
                )
                | qb.userQuery(query_text)
            )
        ),
        "query": query_text,
        "ranking": "fusion",
        "input.query(q)": f"embed({query_text})",
        "hits": top_k,
    }


def atan_norm_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": str(
            qb.select("*")
            .from_(schema_name)
            .where(
                qb.nearestNeighbor(
                    field="embedding",
                    query_vector="q",
                    annotations={"targetHits": 100},
                )
                | qb.userQuery(query_text)
            )
        ),
        "query": query_text,
        "ranking": "atan_norm",
        "input.query(q)": f"embed({query_text})",
        "hits": top_k,
    }
```

def semantic_query_fn(query_text: str, top_k: int) -> dict: return { "yql": str( qb.select("\*") .from\_(schema_name) .where( qb.nearestNeighbor( field="embedding", query_vector="q", annotations={"targetHits": 100}, ) ) ), "query": query_text, "ranking": "semantic", "input.query(q)": f"embed({query_text})", "hits": top_k, } def bm25_query_fn(query_text: str, top_k: int) -> dict: return { "yql": "select * from sources * where userQuery();", # provide the yql directly as a string "query": query_text, "ranking": "bm25", "hits": top_k, } def fusion_query_fn(query_text: str, top_k: int) -> dict: return { "yql": str( qb.select("\*") .from\_(schema_name) .where( qb.nearestNeighbor( field="embedding", query_vector="q", annotations={"targetHits": 100}, ) | qb.userQuery(query_text) ) ), "query": query_text, "ranking": "fusion", "input.query(q)": f"embed({query_text})", "hits": top_k, } def atan_norm_query_fn(query_text: str, top_k: int) -> dict: return { "yql": str( qb.select("\*") .from\_(schema_name) .where( qb.nearestNeighbor( field="embedding", query_vector="q", annotations={"targetHits": 100}, ) | qb.userQuery(query_text) ) ), "query": query_text, "ranking": "atan_norm", "input.query(q)": f"embed({query_text})", "hits": top_k, }

In \[23\]:

Copied!

```
all_results = {}
for evaluator_name, query_fn in [
    ("semantic", semantic_query_fn),
    ("bm25", bm25_query_fn),
    ("fusion", fusion_query_fn),
    ("atan_norm", atan_norm_query_fn),
]:
    print(f"Evaluating {evaluator_name}...")
    evaluator = VespaEvaluator(
        queries=ids_to_query,
        relevant_docs=relevant_docs,
        vespa_query_fn=query_fn,
        app=app,
        name=evaluator_name,
        write_csv=True,  # optionally write metrics to CSV
    )

    results = evaluator.run()
    all_results[evaluator_name] = results
```

all_results = {} for evaluator_name, query_fn in \[ ("semantic", semantic_query_fn), ("bm25", bm25_query_fn), ("fusion", fusion_query_fn), ("atan_norm", atan_norm_query_fn), \]: print(f"Evaluating {evaluator_name}...") evaluator = VespaEvaluator( queries=ids_to_query, relevant_docs=relevant_docs, vespa_query_fn=query_fn, app=app, name=evaluator_name, write_csv=True, # optionally write metrics to CSV ) results = evaluator.run() all_results[evaluator_name] = results

```
Evaluating semantic...
Evaluating bm25...
Evaluating fusion...
Evaluating atan_norm...
```

### Looking at the results[¶](#looking-at-the-results)

In \[24\]:

Copied!

```
results = pd.DataFrame(all_results)
```

results = pd.DataFrame(all_results)

In \[25\]:

Copied!

```
# take out all rows with "searchtime" to a separate dataframe
searchtime = results[results.index.str.contains("searchtime")]
results = results[~results.index.str.contains("searchtime")]


# Highlight the maximum value in each row
def highlight_max(s):
    is_max = s == s.max()
    return ["background-color: lightgreen; color: black;" if v else "" for v in is_max]


# Style the DataFrame: Highlight max values and format numbers to 4 decimals
styled_df = results.style.apply(highlight_max, axis=1).format("{:.4f}")
styled_df
```

# take out all rows with "searchtime" to a separate dataframe

searchtime = results[results.index.str.contains("searchtime")] results = results[~results.index.str.contains("searchtime")]

# Highlight the maximum value in each row

def highlight_max(s): is_max = s == s.max() return ["background-color: lightgreen; color: black;" if v else "" for v in is_max]

# Style the DataFrame: Highlight max values and format numbers to 4 decimals

styled_df = results.style.apply(highlight_max, axis=1).format("{:.4f}") styled_df

Out\[25\]:

|              | semantic | bm25   | fusion | atan_norm |
| ------------ | -------- | ------ | ------ | --------- |
| accuracy@1   | 0.3800   | 0.3000 | 0.4400 | 0.4400    |
| accuracy@3   | 0.6400   | 0.6000 | 0.6800 | 0.7000    |
| accuracy@5   | 0.7200   | 0.6600 | 0.7200 | 0.7400    |
| accuracy@10  | 0.8200   | 0.7400 | 0.8000 | 0.8000    |
| precision@1  | 0.3800   | 0.3000 | 0.4400 | 0.4400    |
| recall@1     | 0.3800   | 0.3000 | 0.4400 | 0.4400    |
| precision@3  | 0.2133   | 0.2000 | 0.2267 | 0.2333    |
| recall@3     | 0.6400   | 0.6000 | 0.6800 | 0.7000    |
| precision@5  | 0.1440   | 0.1320 | 0.1440 | 0.1480    |
| recall@5     | 0.7200   | 0.6600 | 0.7200 | 0.7400    |
| precision@10 | 0.0820   | 0.0740 | 0.0800 | 0.0800    |
| recall@10    | 0.8200   | 0.7400 | 0.8000 | 0.8000    |
| mrr@10       | 0.5309   | 0.4501 | 0.5529 | 0.5738    |
| ndcg@10      | 0.6007   | 0.5206 | 0.6126 | 0.6296    |
| map@100      | 0.5393   | 0.4594 | 0.5630 | 0.5838    |

We can see that for this particular dataset, the hybrid strategy `atan_norm` is the best across all metrics.

In \[26\]:

Copied!

```
results.plot(kind="bar", figsize=(12, 6))
```

results.plot(kind="bar", figsize=(12, 6))

Out\[26\]:

```
<Axes: >
```

### Looking at searchtimes[¶](#looking-at-searchtimes)

Ranking quality is not the only thing that matters. For many applications, search time is equally important.

In \[27\]:

Copied!

```
# plot search time, add (ms) to the y-axis
# convert to ms
searchtime = searchtime * 1000
searchtime.plot(kind="bar", figsize=(12, 6)).set(ylabel="time (ms)")
```

# plot search time, add (ms) to the y-axis

# convert to ms

searchtime = searchtime * 1000 searchtime.plot(kind="bar", figsize=(12, 6)).set(ylabel="time (ms)")

Out\[27\]:

```
[Text(0, 0.5, 'time (ms)')]
```

We can see that both hybrid strategies, `fusion` and `atan_norm` strategy is a bit slower on average than pure `bm25` or `semantic`, as expected.

Depending on the latency budget of your application, this is likely still an attractive trade-off.

## Conclusion and next steps[¶](#conclusion-and-next-steps)

We have shown how you can evaluate a Vespa application on two different levels.

1. Evaluate retrieval (match-phase) using the `VespaMatchEvaluator` class. Here we checked recall, and
1. Evaluate ranking strategies using `VespaEvaluator` class. We have defined and compared 4 different ranking strategies in terms of both ranking quality and searchtime latency.

We hope this can provide you with a good starting point for evaluating your own Vespa application.

If you are ready to advance, you can try to optimize the ranking strategies further, by for example weighing each of the terms in the `atan_norm` strategy differently (`a * normalize_linear(normalized_bm25) + (1-a) * normalize_linear(cos_sim)`) , or by adding a [crossencoder](https://vespa-engine.github.io/pyvespa/examples/cross-encoders-for-global-reranking.md) for re-ranking the top-k results.

## Cleanup[¶](#cleanup)

In \[ \]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()
