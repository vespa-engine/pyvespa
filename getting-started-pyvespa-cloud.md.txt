# Hybrid Search - Quickstart on Vespa Cloud[¶](#hybrid-search-quickstart-on-vespa-cloud)

This is the same guide as [getting-started-pyvespa](https://vespa-engine.github.io/pyvespa/getting-started-pyvespa.md), deploying to Vespa Cloud.

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

**Pre-requisite**: Create a tenant at [cloud.vespa.ai](https://cloud.vespa.ai/), save the tenant name.

## Install[¶](#install)

Install [pyvespa](https://pyvespa.readthedocs.io/) >= 0.45 and the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html). The Vespa CLI is used for data and control plane key management ([Vespa Cloud Security Guide](https://cloud.vespa.ai/en/security/guide)).

In \[ \]:

Copied!

```
!pip3 install pyvespa vespacli
```

!pip3 install pyvespa vespacli

## Configure application[¶](#configure-application)

In \[2\]:

Copied!

```
# Replace with your tenant name from the Vespa Cloud Console
tenant_name = "vespa-team"
# Replace with your application name (does not need to exist yet)
application = "hybridsearch"
```

# Replace with your tenant name from the Vespa Cloud Console

tenant_name = "vespa-team"

# Replace with your application name (does not need to exist yet)

application = "hybridsearch"

## Create an application package[¶](#create-an-application-package)

The [application package](https://vespa-engine.github.io/pyvespa/api/vespa/package.md#vespa.package.ApplicationPackage) has all the Vespa configuration files - create one from scratch:

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
    name=application,
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

from vespa.package import ( ApplicationPackage, Field, Schema, Document, HNSW, RankProfile, Component, Parameter, FieldSet, GlobalPhaseRanking, Function, ) package = ApplicationPackage( name=application, schema=\[ Schema( name="doc", document=Document( fields=\[ Field(name="id", type="string", indexing=["summary"]), Field( name="title", type="string", indexing=["index", "summary"], index="enable-bm25", ), Field( name="body", type="string", indexing=["index", "summary"], index="enable-bm25", bolding=True, ), Field( name="embedding", type="tensor<float>(x[384])", indexing=[ 'input title . " " . input body', "embed", "index", "attribute", ], ann=HNSW(distance_metric="angular"), is_document_field=False, ), \] ), fieldsets=\[FieldSet(name="default", fields=["title", "body"])\], rank_profiles=\[ RankProfile( name="bm25", inputs=\[("query(q)", "tensor<float>(x[384])")\], functions=[ Function(name="bm25sum", expression="bm25(title) + bm25(body)") ], first_phase="bm25sum", ), RankProfile( name="semantic", inputs=\[("query(q)", "tensor<float>(x[384])")\], first_phase="closeness(field, embedding)", ), RankProfile( name="fusion", inherits="bm25", inputs=\[("query(q)", "tensor<float>(x[384])")\], first_phase="closeness(field, embedding)", global_phase=GlobalPhaseRanking( expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))", rerank_count=1000, ), ), \], ) \], components=\[ Component( id="e5", type="hugging-face-embedder", parameters=[ Parameter( "transformer-model", { "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/e5-small-v2-int8.onnx" }, ), Parameter( "tokenizer-model", { "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/tokenizer.json" }, ), ], ) \], )

Note that the name cannot have `-` or `_`.

## Deploy to Vespa Cloud[¶](#deploy-to-vespa-cloud)

The app is now defined and ready to deploy to Vespa Cloud.

Deploy `package` to Vespa Cloud, by creating an instance of [VespaCloud](https://vespa-engine.github.io/pyvespa/api/vespa/deployment.md#vespa.deployment.VespaCloud):

In \[4\]:

Copied!

```
from vespa.deployment import VespaCloud
import os

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

# Key is only used for CI/CD. Can be removed if logging in interactively

key = os.getenv("VESPA_TEAM_API_KEY", None) if key is not None: key = key.replace(r"\\n", "\\n") # To parse key correctly vespa_cloud = VespaCloud( tenant=tenant_name, application=application, key_content=key, # Key is only used for CI/CD. Can be removed if logging in interactively application_package=package, )

```
Setting application...
Running: vespa config set application vespa-team.hybridsearch
Setting target cloud...
Running: vespa config set target cloud

Api-key found for control plane access. Using api-key.
```

For more details on different authentication options and methods, see [authenticating-to-vespa-cloud](https://vespa-engine.github.io/pyvespa/authenticating-to-vespa-cloud.md).

The following will upload the application package to Vespa Cloud Dev Zone (`aws-us-east-1c`), read more about [Vespa Zones](https://cloud.vespa.ai/en/reference/zones.html). The Vespa Cloud Dev Zone is considered as a sandbox environment where resources are down-scaled and idle deployments are expired automatically. For information about production deployments, see the following [method](https://vespa-engine.github.io/pyvespa/api/vespa/deployment.md#vespa.deployment.VespaCloud.deploy_to_prod).

> Note: Deployments to dev and perf expire after 7 days of inactivity, i.e., 7 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 7 more days.

Now deploy the app to Vespa Cloud dev zone.

The first deployment typically takes 2 minutes until the endpoint is up. (Applications that for example refer to large onnx-models may take a bit longer.)

In \[5\]:

Copied!

```
app = vespa_cloud.deploy()
```

app = vespa_cloud.deploy()

```
Deployment started in run 7 of dev-aws-us-east-1c for vespa-team.hybridsearch. This may take a few minutes the first time.
INFO    [07:04:51]  Deploying platform version 8.367.14 and application dev build 6 for dev-aws-us-east-1c of default ...
INFO    [07:04:51]  Using CA signed certificate version 3
INFO    [07:04:52]  Using 1 nodes in container cluster 'hybridsearch_container'
INFO    [07:04:53]  Validating Onnx models memory usage for container cluster 'hybridsearch_container', percentage of available memory too low (10 < 15) to avoid restart, consider a flavor with more memory to avoid this
WARNING [07:04:53]  Auto-overriding validation which would be disallowed in production: certificate-removal: Data plane certificate(s) from cluster 'hybridsearch_container' is removed (removed certificates: [CN=cloud.vespa.example]) This can cause client connection issues.. To allow this add <allow until='yyyy-mm-dd'>certificate-removal</allow> to validation-overrides.xml, see https://docs.vespa.ai/en/reference/validation-overrides.html
INFO    [07:04:55]  Session 298587 for tenant 'vespa-team' prepared and activated.
INFO    [07:04:55]  ######## Details for all nodes ########
INFO    [07:04:55]  h94416a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [07:04:55]  --- platform vespa/cloud-tenant-rhel8:8.367.14
INFO    [07:04:55]  --- container on port 4080 has config generation 298580, wanted is 298587
INFO    [07:04:55]  --- metricsproxy-container on port 19092 has config generation 298587, wanted is 298587
INFO    [07:04:55]  h94249f.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [07:04:55]  --- platform vespa/cloud-tenant-rhel8:8.367.14
INFO    [07:04:55]  --- container-clustercontroller on port 19050 has config generation 298580, wanted is 298587
INFO    [07:04:55]  --- metricsproxy-container on port 19092 has config generation 298580, wanted is 298587
INFO    [07:04:55]  h93394a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [07:04:55]  --- platform vespa/cloud-tenant-rhel8:8.367.14
INFO    [07:04:55]  --- logserver-container on port 4080 has config generation 298587, wanted is 298587
INFO    [07:04:55]  --- metricsproxy-container on port 19092 has config generation 298580, wanted is 298587
INFO    [07:04:55]  h94419a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [07:04:55]  --- platform vespa/cloud-tenant-rhel8:8.367.14
INFO    [07:04:55]  --- storagenode on port 19102 has config generation 298587, wanted is 298587
INFO    [07:04:55]  --- searchnode on port 19107 has config generation 298587, wanted is 298587
INFO    [07:04:55]  --- distributor on port 19111 has config generation 298587, wanted is 298587
INFO    [07:04:55]  --- metricsproxy-container on port 19092 has config generation 298587, wanted is 298587
INFO    [07:05:02]  Found endpoints:
INFO    [07:05:02]  - dev.aws-us-east-1c
INFO    [07:05:02]   |-- https://f7f73182.eb1181f2.z.vespa-app.cloud/ (cluster 'hybridsearch_container')
INFO    [07:05:02]  Deployment of new application complete!
Found mtls endpoint for hybridsearch_container
URL: https://f7f73182.eb1181f2.z.vespa-app.cloud/
Connecting to https://f7f73182.eb1181f2.z.vespa-app.cloud/
Using mtls_key_cert Authentication against endpoint https://f7f73182.eb1181f2.z.vespa-app.cloud//ApplicationStatus
Application is up!
Finished deployment.
```

If the deployment failed, it is possible you forgot to add the key in the Vespa Cloud Console in the `vespa auth api-key` step above.

If you can authenticate, you should see lines like the following

```
 Deployment started in run 1 of dev-aws-us-east-1c for mytenant.hybridsearch.
```

The deployment takes a few minutes the first time while Vespa Cloud sets up the resources for your Vespa application

`app` now holds a reference to a [Vespa](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa) instance. We can access the mTLS protected endpoint name using the control-plane (vespa_cloud) instance. This endpoint we can query and feed to (data plane access) using the mTLS certificate generated in previous steps.

In \[6\]:

Copied!

```
endpoint = vespa_cloud.get_mtls_endpoint()
endpoint
```

endpoint = vespa_cloud.get_mtls_endpoint() endpoint

```
Found mtls endpoint for hybridsearch_container
URL: https://f7f73182.eb1181f2.z.vespa-app.cloud/
```

Out\[6\]:

```
'https://f7f73182.eb1181f2.z.vespa-app.cloud/'
```

## Feeding documents to Vespa[¶](#feeding-documents-to-vespa)

In this example we use the [HF Datasets](https://huggingface.co/docs/datasets/index) library to stream the [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) dataset and index in our newly deployed Vespa instance. Read more about the [NFCorpus](https://huggingface.co/datasets/mteb/nfcorpus):

> NFCorpus is a full-text English retrieval data set for Medical Information Retrieval.

The following uses the [stream](https://huggingface.co/docs/datasets/stream) option of datasets to stream the data without downloading all the contents locally. The `map` functionality allows us to convert the dataset fields into the expected feed format for `pyvespa` which expects a dict with the keys `id` and `fields`:

`{ "id": "vespa-document-id", "fields": {"vespa_field": "vespa-field-value"}}`

In \[7\]:

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

Now we can feed to Vespa using `feed_iterable` which accepts any `Iterable` and an optional callback function where we can check the outcome of each operation. The application is configured to use [embedding](https://docs.vespa.ai/en/embedding.html) functionality, that produce a vector embedding using a concatenation of the title and the body input fields. This step is resource intensive.

Read more about embedding inference in Vespa in the [Accelerating Transformer-based Embedding Retrieval with Vespa](https://blog.vespa.ai/accelerating-transformer-based-embedding-retrieval-with-vespa/) blog post.

Default node resources in Vespa Cloud have 2 v-cpu for the Dev Zone.

In \[8\]:

Copied!

```
from vespa.io import VespaResponse, VespaQueryResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Error when feeding document {id}: {response.get_json()}")


app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)
```

from vespa.io import VespaResponse, VespaQueryResponse def callback(response: VespaResponse, id: str): if not response.is_successful(): print(f"Error when feeding document {id}: {response.get_json()}") app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)

```
Using mtls_key_cert Authentication against endpoint https://f7f73182.eb1181f2.z.vespa-app.cloud//ApplicationStatus
```

## Querying Vespa[¶](#querying-vespa)

Using the [Vespa Query language](https://docs.vespa.ai/en/query-language.html) we can query the indexed data.

- Using a context manager `with app.syncio() as session` to handle connection pooling ([best practices](https://cloud.vespa.ai/en/http-best-practices))
- The query method accepts any valid Vespa [query api parameter](https://docs.vespa.ai/en/reference/query-api-reference.html) in `**kwargs`
- Vespa api parameter names that contains `.` must be sent as `dict` parameters in the `body` method argument

The following searches for `How Fruits and Vegetables Can Treat Asthma?` using different retrieval and [ranking](https://docs.vespa.ai/en/ranking.html) strategies.

Query the text search app using the [Vespa Query language](https://docs.vespa.ai/en/query-language.html) by sending the parameters to the body argument of [Vespa.query](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa.query).

First we define a simple routine that will return a dataframe of the results for prettier display in the notebook.

In \[9\]:

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

In \[10\]:

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

```
         id                                              title
0  MED-2450  Protective effect of fruits, vegetables and th...
1  MED-2464  Low vegetable intake is associated with allerg...
2  MED-1162  Pesticide residues in imported, organic, and "...
3  MED-2461  The association of diet with respiratory sympt...
4  MED-2085  Antiplatelet, anticoagulant, and fibrinolytic ...
```

### Plain Semantic Search[¶](#plain-semantic-search)

The following uses dense vector representations of the query and the document and matching is performed and accelerated by Vespa's support for [approximate nearest neighbor search](https://docs.vespa.ai/en/approximate-nn-hnsw.html). The vector embedding representation of the text is obtained using Vespa's [embedder functionality](https://docs.vespa.ai/en/embedding.html#embedding-a-query-text).

In \[11\]:

Copied!

```
with app.syncio(connections=1) as session:
    query = "How Fruits and Vegetables Can Treat Asthma?"
    response: VespaQueryResponse = session.query(
        yql="select * from sources * where ({targetHits:5}nearestNeighbor(embedding,q)) limit 5",
        query=query,
        ranking="semantic",
        body={"input.query(q)": f"embed({query})"},
    )
    assert response.is_successful()
    print(display_hits_as_df(response, ["id", "title"]))
```

with app.syncio(connections=1) as session: query = "How Fruits and Vegetables Can Treat Asthma?" response: VespaQueryResponse = session.query( yql="select * from sources * where ({targetHits:5}nearestNeighbor(embedding,q)) limit 5", query=query, ranking="semantic", body={"input.query(q)": f"embed({query})"}, ) assert response.is_successful() print(display_hits_as_df(response, ["id", "title"]))

```
         id                                              title
0  MED-5072  Lycopene-rich treatments modify noneosinophili...
1  MED-2472  Vegan regimen with reduced medication in the t...
2  MED-2464  Low vegetable intake is associated with allerg...
3  MED-2458  Manipulating antioxidant intake in asthma: a r...
4  MED-2450  Protective effect of fruits, vegetables and th...
```

### Hybrid Search[¶](#hybrid-search)

This is one approach to combine the two retrieval strategies and where we use Vespa's support for [cross-hits feature normalization and reciprocal rank fusion](https://docs.vespa.ai/en/phased-ranking.html#cross-hit-normalization-including-reciprocal-rank-fusion). This functionality is exposed in the context of `global` re-ranking, after the distributed query retrieval execution which might span 1000s of nodes.

#### Hybrid search with the OR query operator[¶](#hybrid-search-with-the-or-query-operator)

This combines the two methods using logical disjunction (OR). Note that the first-phase expression in our `fusion` expression is only using the semantic score, this because usually semantic search provides better recall than sparse keyword search alone.

In \[12\]:

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

```
         id                                              title
0  MED-2464  Low vegetable intake is associated with allerg...
1  MED-2450  Protective effect of fruits, vegetables and th...
2  MED-2458  Manipulating antioxidant intake in asthma: a r...
3  MED-2461  The association of diet with respiratory sympt...
4  MED-5072  Lycopene-rich treatments modify noneosinophili...
```

#### Hybrid search with the RANK query operator[¶](#hybrid-search-with-the-rank-query-operator)

This combines the two methods using the [rank](https://docs.vespa.ai/en/reference/query-language-reference.html#rank) query operator. In this case we express that we want to retrieve the top-1000 documents using vector search, and then have sparse features like BM25 calculated as well (second operand of the rank operator). Finally the hits are re-ranked using the reciprocal rank fusion

In \[13\]:

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

```
         id                                              title
0  MED-2464  Low vegetable intake is associated with allerg...
1  MED-2450  Protective effect of fruits, vegetables and th...
2  MED-2458  Manipulating antioxidant intake in asthma: a r...
3  MED-2461  The association of diet with respiratory sympt...
4  MED-5072  Lycopene-rich treatments modify noneosinophili...
```

#### Hybrid search with filters[¶](#hybrid-search-with-filters)

In this example we add another query term to the yql, restricting the nearest neighbor search to only consider documents that have vegetable in the title.

In \[14\]:

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

```
         id                                              title
0  MED-2464  Low vegetable intake is associated with allerg...
1  MED-2450  Protective effect of fruits, vegetables and th...
2  MED-3199  Potential risks resulting from fruit/vegetable...
3  MED-2085  Antiplatelet, anticoagulant, and fibrinolytic ...
4  MED-4496  The effect of fruit and vegetable intake on ri...
```

## Next steps[¶](#next-steps)

This is just an intro into the capabilities of Vespa and pyvespa. Browse the site to learn more about schemas, feeding and queries - find more complex applications in [examples](https://vespa-engine.github.io/pyvespa/examples).

## Example: Document operations using cert/key pair[¶](#example-document-operations-using-certkey-pair)

Above, we deployed to Vespa Cloud, and as part of that, generated a data-plane mTLS cert/key pair.

This pair can be used to access the dataplane for reads/writes to documents and running queries from many different clients. The following demonstrates that using the `requests` library.

Set up a dataplane connection using the cert/key pair:

In \[15\]:

Copied!

```
import requests

cert_path = app.cert
key_path = app.key
session = requests.Session()
session.cert = (cert_path, key_path)
```

import requests cert_path = app.cert key_path = app.key session = requests.Session() session.cert = (cert_path, key_path)

Get a document from the endpoint returned when we deployed to Vespa Cloud above. PyVespa wraps the Vespa [document api](https://docs.vespa.ai/en/document-v1-api-guide.html) internally and in these examples we use the document api directly, but with the mTLS key/cert pair we used when deploying the app.

In \[16\]:

Copied!

```
url = "{0}/document/v1/{1}/{2}/docid/{3}".format(endpoint, "tutorial", "doc", "MED-10")
doc = session.get(url).json()
doc
```

url = "{0}/document/v1/{1}/{2}/docid/{3}".format(endpoint, "tutorial", "doc", "MED-10") doc = session.get(url).json() doc

Out\[16\]:

```
{'pathId': '/document/v1/tutorial/doc/docid/MED-10',
 'id': 'id:tutorial:doc::MED-10',
 'fields': {'body': 'Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear. We evaluated risk of breast cancer death among statin users in a population-based cohort of breast cancer patients. The study cohort included all newly diagnosed breast cancer patients in Finland during 1995–2003 (31,236 cases), identified from the Finnish Cancer Registry. Information on statin use before and after the diagnosis was obtained from a national prescription database. We used the Cox proportional hazards regression method to estimate mortality among statin users with statin use as time-dependent variable. A total of 4,151 participants had used statins. During the median follow-up of 3.25 years after the diagnosis (range 0.08–9.0 years) 6,011 participants died, of which 3,619 (60.2%) was due to breast cancer. After adjustment for age, tumor characteristics, and treatment selection, both post-diagnostic and pre-diagnostic statin use were associated with lowered risk of breast cancer death (HR 0.46, 95% CI 0.38–0.55 and HR 0.54, 95% CI 0.44–0.67, respectively). The risk decrease by post-diagnostic statin use was likely affected by healthy adherer bias; that is, the greater likelihood of dying cancer patients to discontinue statin use as the association was not clearly dose-dependent and observed already at low-dose/short-term use. The dose- and time-dependence of the survival benefit among pre-diagnostic statin users suggests a possible causal effect that should be evaluated further in a clinical trial testing statins’ effect on survival in breast cancer patients.',
  'title': 'Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland',
  'id': 'MED-10'}}
```

Update the title and post the new version:

In \[17\]:

Copied!

```
doc["fields"]["title"] = "Can you eat lobster?"
response = session.post(url, json=doc).json()
response
```

doc["fields"]["title"] = "Can you eat lobster?" response = session.post(url, json=doc).json() response

Out\[17\]:

```
{'pathId': '/document/v1/tutorial/doc/docid/MED-10',
 'id': 'id:tutorial:doc::MED-10'}
```

Get the doc again to see the new title:

In \[18\]:

Copied!

```
doc = session.get(url).json()
doc
```

doc = session.get(url).json() doc

Out\[18\]:

```
{'pathId': '/document/v1/tutorial/doc/docid/MED-10',
 'id': 'id:tutorial:doc::MED-10',
 'fields': {'body': 'Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear. We evaluated risk of breast cancer death among statin users in a population-based cohort of breast cancer patients. The study cohort included all newly diagnosed breast cancer patients in Finland during 1995–2003 (31,236 cases), identified from the Finnish Cancer Registry. Information on statin use before and after the diagnosis was obtained from a national prescription database. We used the Cox proportional hazards regression method to estimate mortality among statin users with statin use as time-dependent variable. A total of 4,151 participants had used statins. During the median follow-up of 3.25 years after the diagnosis (range 0.08–9.0 years) 6,011 participants died, of which 3,619 (60.2%) was due to breast cancer. After adjustment for age, tumor characteristics, and treatment selection, both post-diagnostic and pre-diagnostic statin use were associated with lowered risk of breast cancer death (HR 0.46, 95% CI 0.38–0.55 and HR 0.54, 95% CI 0.44–0.67, respectively). The risk decrease by post-diagnostic statin use was likely affected by healthy adherer bias; that is, the greater likelihood of dying cancer patients to discontinue statin use as the association was not clearly dose-dependent and observed already at low-dose/short-term use. The dose- and time-dependence of the survival benefit among pre-diagnostic statin users suggests a possible causal effect that should be evaluated further in a clinical trial testing statins’ effect on survival in breast cancer patients.',
  'title': 'Can you eat lobster?',
  'id': 'MED-10'}}
```

## Example: Reconnect pyvespa using cert/key pair[¶](#example-reconnect-pyvespa-using-certkey-pair)

Above, we stored the dataplane credentials for later use. Deployment of an application usually happens when the schema changes, whereas accessing the dataplane is for document updates and user queries.

One only needs to know the endpoint and the cert/key pair to enable a connection to a Vespa Cloud application:

In \[19\]:

Copied!

```
# cert_path = "/Users/me/.vespa/mytenant.hybridsearch.default/data-plane-public-cert.pem"
# key_path  = "/Users/me/.vespa/mytenant.hybridsearch.default/data-plane-private-key.pem"

from vespa.application import Vespa

the_app = Vespa(endpoint, cert=cert_path, key=key_path)

res = the_app.query(
    yql="select documentid, id, title from sources * where userQuery()",
    query="Can you eat lobster?",
    ranking="bm25",
)
res.hits[0]
```

# cert_path = "/Users/me/.vespa/mytenant.hybridsearch.default/data-plane-public-cert.pem"

# key_path = "/Users/me/.vespa/mytenant.hybridsearch.default/data-plane-private-key.pem"

from vespa.application import Vespa the_app = Vespa(endpoint, cert=cert_path, key=key_path) res = the_app.query( yql="select documentid, id, title from sources * where userQuery()", query="Can you eat lobster?", ranking="bm25", ) res.hits[0]

```
Using mtls_key_cert Authentication against endpoint https://f7f73182.eb1181f2.z.vespa-app.cloud//ApplicationStatus
```

Out\[19\]:

```
{'id': 'id:tutorial:doc::MED-10',
 'relevance': 25.27992205160453,
 'source': 'hybridsearch_content',
 'fields': {'documentid': 'id:tutorial:doc::MED-10',
  'id': 'MED-10',
  'title': 'Can you eat lobster?'}}
```

A common problem is a cert mismatch - the cert/key pair used when deployed is different than the pair used when making requests against Vespa. This will cause 40x errors.

Make sure it is the same pair / re-create with `vespa auth cert -f` AND redeploy.

If you re-generate a mTLS certificate pair, and use that when connecting to Vespa cloud endpoint, it will fail until you have updaded the deployment with the new public certificate.

### Delete application[¶](#delete-application)

The following will delete the application and data from the dev environment.

In \[20\]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()

```
Deactivated vespa-team.hybridsearch in dev.aws-us-east-1c
Deleted instance vespa-team.hybridsearch.default
```
