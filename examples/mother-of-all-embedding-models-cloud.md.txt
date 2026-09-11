# BGE-M3 - The Mother of all embedding models[¶](#bge-m3-the-mother-of-all-embedding-models)

BAAI released BGE-M3 on January 30th, a new member of the BGE model series.

> M3 stands for Multi-linguality (100+ languages), Multi-granularities (input length up to 8192), Multi-Functionality (unification of dense, lexical, multi-vec (colbert) retrieval).

This notebook demonstrates how to use the [BGE-M3](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/BGE_M3.pdf) embeddings and represent all three embedding representations in Vespa! Vespa is the only scalable serving engine that can handle all M3 representations.

This code is inspired by the README from the model hub [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3).

Let's get started! First, install dependencies:

In \[ \]:

Copied!

```
!pip3 install -U pyvespa FlagEmbedding vespacli
```

!pip3 install -U pyvespa FlagEmbedding vespacli

### Explore the multiple representations of M3[¶](#explore-the-multiple-representations-of-m3)

When encoding text, we can ask for the representations we want

- Sparse vectors with weights for the token IDs (from the multilingual tokenization process)
- Dense (DPR) regular text embeddings
- Multi-Dense (ColBERT) - contextualized multi-token vectors

Let us dive into it - To use this model on the CPU we set `use_fp16` to False, for GPU inference, it is recommended to use `use_fp16=True` for accelerated inference.

In \[ \]:

Copied!

```
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
```

from FlagEmbedding import BGEM3FlagModel model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

## A demo passage[¶](#a-demo-passage)

Let us encode a simple passage

In \[3\]:

Copied!

```
passage = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."
]
```

passage = [ "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction." ]

In \[ \]:

Copied!

```
passage_embeddings = model.encode(
    passage, return_dense=True, return_sparse=True, return_colbert_vecs=True
)
```

passage_embeddings = model.encode( passage, return_dense=True, return_sparse=True, return_colbert_vecs=True )

In \[5\]:

Copied!

```
passage_embeddings.keys()
```

passage_embeddings.keys()

Out\[5\]:

```
dict_keys(['dense_vecs', 'lexical_weights', 'colbert_vecs'])
```

## Defining the Vespa application[¶](#defining-the-vespa-application)

[PyVespa](https://vespa-engine.github.io/pyvespa/) helps us build the [Vespa application package](https://docs.vespa.ai/en/application-packages.html). A Vespa application package consists of configuration files, schemas, models, and code (plugins).

First, we define a [Vespa schema](https://docs.vespa.ai/en/schemas.html) with the fields we want to store and their type. We use Vespa [tensors](https://docs.vespa.ai/en/tensor-user-guide.html) to represent the three different M3 representations.

- We use a mapped tensor denoted by `t{}` to represent the sparse lexical representation
- We use an indexed tensor denoted by `x[1024]` to represent the dense single vector representation of 1024 dimensions
- For the colbert_rep (multi-vector), we use a mixed tensor that combines a mapped and an indexed dimension. This mixed tensor allows us to represent variable lengths.

We use `bfloat16` tensor cell type, saving 50% storage compared to `float`.

In \[6\]:

Copied!

```
from vespa.package import Schema, Document, Field, FieldSet

m_schema = Schema(
    name="m",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary"]),
            Field(
                name="text",
                type="string",
                indexing=["summary", "index"],
                index="enable-bm25",
            ),
            Field(
                name="lexical_rep",
                type="tensor<bfloat16>(t{})",
                indexing=["summary", "attribute"],
            ),
            Field(
                name="dense_rep",
                type="tensor<bfloat16>(x[1024])",
                indexing=["summary", "attribute"],
                attribute=["distance-metric: angular"],
            ),
            Field(
                name="colbert_rep",
                type="tensor<bfloat16>(t{}, x[1024])",
                indexing=["summary", "attribute"],
            ),
        ],
    ),
    fieldsets=[FieldSet(name="default", fields=["text"])],
)
```

from vespa.package import Schema, Document, Field, FieldSet m_schema = Schema( name="m", document=Document( fields=\[ Field(name="id", type="string", indexing=["summary"]), Field( name="text", type="string", indexing=["summary", "index"], index="enable-bm25", ), Field( name="lexical_rep", type="tensor<bfloat16>(t{})", indexing=["summary", "attribute"], ), Field( name="dense_rep", type="tensor<bfloat16>(x[1024])", indexing=["summary", "attribute"], attribute=["distance-metric: angular"], ), Field( name="colbert_rep", type="tensor<bfloat16>(t{}, x[1024])", indexing=["summary", "attribute"], ), \], ), fieldsets=\[FieldSet(name="default", fields=["text"])\], )

The above defines our `m` schema with the original text and the three different representations

In \[7\]:

Copied!

```
from vespa.package import ApplicationPackage

vespa_app_name = "m"
vespa_application_package = ApplicationPackage(name=vespa_app_name, schema=[m_schema])
```

from vespa.package import ApplicationPackage vespa_app_name = "m" vespa_application_package = ApplicationPackage(name=vespa_app_name, schema=[m_schema])

In the last step, we configure [ranking](https://docs.vespa.ai/en/ranking.html) by adding `rank-profile`'s to the schema.

We define three functions that implement the three different scoring functions for the different representations

- dense (dense cosine similarity)
- sparse (sparse dot product)
- max_sim (The colbert max sim operation)

Then, we combine these three scoring functions using a linear combination with weights, as suggested by the authors [here](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/BGE_M3.pdf#compute-score-for-text-pairs).

In \[8\]:

Copied!

```
from vespa.package import RankProfile, Function, FirstPhaseRanking


semantic = RankProfile(
    name="m3hybrid",
    inputs=[
        ("query(q_dense)", "tensor<bfloat16>(x[1024])"),
        ("query(q_lexical)", "tensor<bfloat16>(t{})"),
        ("query(q_colbert)", "tensor<bfloat16>(qt{}, x[1024])"),
        ("query(q_len_colbert)", "float"),
    ],
    functions=[
        Function(
            name="dense",
            expression="cosine_similarity(query(q_dense), attribute(dense_rep),x)",
        ),
        Function(
            name="lexical", expression="sum(query(q_lexical) * attribute(lexical_rep))"
        ),
        Function(
            name="max_sim",
            expression="sum(reduce(sum(query(q_colbert) * attribute(colbert_rep) , x),max, t),qt)/query(q_len_colbert)",
        ),
    ],
    first_phase=FirstPhaseRanking(
        expression="0.4*dense + 0.2*lexical +  0.4*max_sim", rank_score_drop_limit=0.0
    ),
    match_features=["dense", "lexical", "max_sim", "bm25(text)"],
)
m_schema.add_rank_profile(semantic)
```

from vespa.package import RankProfile, Function, FirstPhaseRanking semantic = RankProfile( name="m3hybrid", inputs=\[ ("query(q_dense)", "tensor<bfloat16>(x[1024])"), ("query(q_lexical)", "tensor<bfloat16>(t{})"), ("query(q_colbert)", "tensor<bfloat16>(qt{}, x[1024])"), ("query(q_len_colbert)", "float"), \], functions=[ Function( name="dense", expression="cosine_similarity(query(q_dense), attribute(dense_rep),x)", ), Function( name="lexical", expression="sum(query(q_lexical) * attribute(lexical_rep))" ), Function( name="max_sim", expression="sum(reduce(sum(query(q_colbert) * attribute(colbert_rep) , x),max, t),qt)/query(q_len_colbert)", ), ], first_phase=FirstPhaseRanking( expression="0.4\*dense + 0.2\*lexical + 0.4\*max_sim", rank_score_drop_limit=0.0 ), match_features=["dense", "lexical", "max_sim", "bm25(text)"], ) m_schema.add_rank_profile(semantic)

The `m3hybrid` rank-profile above defines the query input embedding type and a similarities function that uses a Vespa [tensor compute function](https://docs.vespa.ai/en/reference/ranking-expressions.html#tensor-functions) that calculates the M3 similarities for dense, lexical, and the max_sim for the colbert representations.

The profile only defines a single ranking phase, using a linear combination of multiple features using the suggested weighting.

Using [match-features](https://docs.vespa.ai/en/reference/schema-reference.html#match-features), Vespa returns selected features along with the hit in the SERP (result page). We also include BM25. We can view BM25 as the fourth dimension. Especially for long-context retrieval, it can be helpful compared to the neural representations.

## Deploy the application to Vespa Cloud[¶](#deploy-the-application-to-vespa-cloud)

With the configured application, we can deploy it to [Vespa Cloud](https://cloud.vespa.ai/en/).

To deploy the application to Vespa Cloud we need to create a tenant in the Vespa Cloud:

Create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/) (unless you already have one). This step requires a Google or GitHub account, and will start your [free trial](https://cloud.vespa.ai/en/free-trial).

Make note of the tenant name, it is used in the next steps.

> Note: Deployments to dev and perf expire after 7 days of inactivity, i.e., 7 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 7 more days.

In \[13\]:

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
    application=vespa_app_name,
    key_content=key,  # Key is only used for CI/CD. Can be removed if logging in interactively
    application_package=vespa_application_package,
)
```

from vespa.deployment import VespaCloud import os

# Replace with your tenant name from the Vespa Cloud Console

tenant_name = "vespa-team"

# Key is only used for CI/CD. Can be removed if logging in interactively

key = os.getenv("VESPA_TEAM_API_KEY", None) if key is not None: key = key.replace(r"\\n", "\\n") # To parse key correctly vespa_cloud = VespaCloud( tenant=tenant_name, application=vespa_app_name, key_content=key, # Key is only used for CI/CD. Can be removed if logging in interactively application_package=vespa_application_package, )

Now deploy the app to Vespa Cloud dev zone.

The first deployment typically takes 2 minutes until the endpoint is up.

In \[14\]:

Copied!

```
from vespa.application import Vespa

app: Vespa = vespa_cloud.deploy()
```

from vespa.application import Vespa app: Vespa = vespa_cloud.deploy()

```
Deployment started in run 1 of dev-aws-us-east-1c for samples.m. This may take a few minutes the first time.
INFO    [22:13:09]  Deploying platform version 8.299.14 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [22:13:10]  Using CA signed certificate version 0
INFO    [22:13:10]  Using 1 nodes in container cluster 'm_container'
INFO    [22:13:14]  Session 939 for tenant 'samples' prepared and activated.
INFO    [22:13:17]  ######## Details for all nodes ########
INFO    [22:13:31]  h88976d.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [22:13:31]  --- platform vespa/cloud-tenant-rhel8:8.299.14 <-- :
INFO    [22:13:31]  --- container-clustercontroller on port 19050 has not started 
INFO    [22:13:31]  --- metricsproxy-container on port 19092 has not started 
INFO    [22:13:31]  h89388b.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [22:13:31]  --- platform vespa/cloud-tenant-rhel8:8.299.14 <-- :
INFO    [22:13:31]  --- storagenode on port 19102 has not started 
INFO    [22:13:31]  --- searchnode on port 19107 has not started 
INFO    [22:13:31]  --- distributor on port 19111 has not started 
INFO    [22:13:31]  --- metricsproxy-container on port 19092 has not started 
INFO    [22:13:31]  h90001a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [22:13:31]  --- platform vespa/cloud-tenant-rhel8:8.299.14 <-- :
INFO    [22:13:31]  --- logserver-container on port 4080 has not started 
INFO    [22:13:31]  --- metricsproxy-container on port 19092 has not started 
INFO    [22:13:31]  h90550a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [22:13:31]  --- platform vespa/cloud-tenant-rhel8:8.299.14 <-- :
INFO    [22:13:31]  --- container on port 4080 has not started 
INFO    [22:13:31]  --- metricsproxy-container on port 19092 has not started 
INFO    [22:14:31]  Found endpoints:
INFO    [22:14:31]  - dev.aws-us-east-1c
INFO    [22:14:31]   |-- https://d29bf3e7.f064e220.z.vespa-app.cloud/ (cluster 'm_container')
INFO    [22:14:32]  Installation succeeded!
Using mTLS (key,cert) Authentication against endpoint https://d29bf3e7.f064e220.z.vespa-app.cloud//ApplicationStatus
Application is up!
Finished deployment.
```

## Feed the M3 representations[¶](#feed-the-m3-representations)

We convert the three different representations to Vespa feed format

In \[15\]:

Copied!

```
vespa_fields = {
    "text": passage[0],
    "lexical_rep": {
        key: float(value)
        for key, value in passage_embeddings["lexical_weights"][0].items()
    },
    "dense_rep": passage_embeddings["dense_vecs"][0].tolist(),
    "colbert_rep": {
        index: passage_embeddings["colbert_vecs"][0][index].tolist()
        for index in range(passage_embeddings["colbert_vecs"][0].shape[0])
    },
}
```

vespa_fields = { "text": passage[0], "lexical_rep": { key: float(value) for key, value in passage_embeddings["lexical_weights"][0].items() }, "dense_rep": passage_embeddings["dense_vecs"][0].tolist(), "colbert_rep": { index: passage_embeddings["colbert_vecs"][0][index].tolist() for index in range(passage_embeddings["colbert_vecs"][0].shape[0]) }, }

In \[17\]:

Copied!

```
from vespa.io import VespaResponse

response: VespaResponse = app.feed_data_point(
    schema="m", data_id=0, fields=vespa_fields
)
assert response.is_successful()
```

from vespa.io import VespaResponse response: VespaResponse = app.feed_data_point( schema="m", data_id=0, fields=vespa_fields ) assert response.is_successful()

### Querying data[¶](#querying-data)

Now, we can also query our data.

Read more about querying Vespa in:

- [Vespa Query API](https://docs.vespa.ai/en/query-api.html)
- [Vespa Query API reference](https://docs.vespa.ai/en/reference/query-api-reference.html)
- [Vespa Query Language API (YQL)](https://docs.vespa.ai/en/query-language.html)

In \[ \]:

Copied!

```
query = ["What is BGE M3?"]
query_embeddings = model.encode(
    query, return_dense=True, return_sparse=True, return_colbert_vecs=True
)
```

query = ["What is BGE M3?"] query_embeddings = model.encode( query, return_dense=True, return_sparse=True, return_colbert_vecs=True )

The M3 colbert scoring function needs the query length to normalize the score to the range 0 to 1. This helps when combining the score with the other scoring functions.

In \[19\]:

Copied!

```
query_length = query_embeddings["colbert_vecs"][0].shape[0]
```

query_length = query_embeddings["colbert_vecs"][0].shape[0]

In \[20\]:

Copied!

```
query_fields = {
    "input.query(q_lexical)": {
        key: float(value)
        for key, value in query_embeddings["lexical_weights"][0].items()
    },
    "input.query(q_dense)": query_embeddings["dense_vecs"][0].tolist(),
    "input.query(q_colbert)": str(
        {
            index: query_embeddings["colbert_vecs"][0][index].tolist()
            for index in range(query_embeddings["colbert_vecs"][0].shape[0])
        }
    ),
    "input.query(q_len_colbert)": query_length,
}
```

query_fields = { "input.query(q_lexical)": { key: float(value) for key, value in query_embeddings["lexical_weights"][0].items() }, "input.query(q_dense)": query_embeddings["dense_vecs"][0].tolist(), "input.query(q_colbert)": str( { index: query_embeddings["colbert_vecs"][0][index].tolist() for index in range(query_embeddings["colbert_vecs"][0].shape[0]) } ), "input.query(q_len_colbert)": query_length, }

In \[21\]:

Copied!

```
from vespa.io import VespaQueryResponse
import json

response: VespaQueryResponse = app.query(
    yql="select id, text from m where userQuery() or ({targetHits:10}nearestNeighbor(dense_rep,q_dense))",
    ranking="m3hybrid",
    query=query[0],
    body={**query_fields},
)
assert response.is_successful()
print(json.dumps(response.hits[0], indent=2))
```

from vespa.io import VespaQueryResponse import json response: VespaQueryResponse = app.query( yql="select id, text from m where userQuery() or ({targetHits:10}nearestNeighbor(dense_rep,q_dense))", ranking="m3hybrid", query=query[0], body={\*\*query_fields}, ) assert response.is_successful() print(json.dumps(response.hits[0], indent=2))

```
{
  "id": "index:m_content/0/cfcd2084234135f700f08abf",
  "relevance": 0.5993361056332731,
  "source": "m_content",
  "fields": {
    "matchfeatures": {
      "bm25(text)": 0.8630462173553426,
      "dense": 0.6258970723760484,
      "lexical": 0.1941967010498047,
      "max_sim": 0.7753448411822319
    },
    "text": "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."
  }
}
```

Notice the `matchfeatures` that returns the configured match-features from the rank-profile. We can use these to compare the torch model scoring with the computations specified in Vespa.

Now, we can compare the Vespa computed scores with the model torch code and they line up perfectly

In \[22\]:

Copied!

```
model.compute_lexical_matching_score(
    passage_embeddings["lexical_weights"][0], query_embeddings["lexical_weights"][0]
)
```

model.compute_lexical_matching_score( passage_embeddings["lexical_weights"][0], query_embeddings["lexical_weights"][0] )

Out\[22\]:

```
0.19554455392062664
```

In \[23\]:

Copied!

```
query_embeddings["dense_vecs"][0] @ passage_embeddings["dense_vecs"][0].T
```

query_embeddings["dense_vecs"][0] @ passage_embeddings["dense_vecs"][0].T

Out\[23\]:

```
0.6259037
```

In \[24\]:

Copied!

```
model.colbert_score(
    query_embeddings["colbert_vecs"][0], passage_embeddings["colbert_vecs"][0]
)
```

model.colbert_score( query_embeddings["colbert_vecs"][0], passage_embeddings["colbert_vecs"][0] )

Out\[24\]:

```
tensor(0.7797)
```

### That is it![¶](#that-is-it)

That is how easy it is to represent the brand new M3 FlagEmbedding representations in Vespa! Read more in the [M3 technical report](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/BGE_M3.pdf).

We can go ahead and delete the Vespa cloud instance we deployed by:

In \[ \]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()
