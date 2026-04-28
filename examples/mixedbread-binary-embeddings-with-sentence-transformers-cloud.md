# Using Mixedbread.ai embedding model with support for binary vectors[¶](#using-mixedbreadai-embedding-model-with-support-for-binary-vectors)

Check out the amazing blog post: [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization)

Binarization is significant because:

- Binarization reduces the storage footprint from 1024 floats (4096 bytes) per vector to 128 int8 (128 bytes).
- 32x less data to store
- Faster distance calculations using [hamming](https://docs.vespa.ai/en/reference/schema-reference.html#distance-metric) distance, which Vespa natively supports for bits packed into int8 precision. More on [hamming distance in Vespa](https://docs.vespa.ai/en/reference/schema-reference.html#hamming).

Vespa supports `hamming` distance with and without [hnsw indexing](https://docs.vespa.ai/en/approximate-nn-hnsw.html).

For those wanting to learn more about binary vectors, we recommend our 2021 blog series on [Billion-scale vector search with Vespa](https://blog.vespa.ai/billion-scale-knn/) and [Billion-scale vector search with Vespa - part two](https://blog.vespa.ai/billion-scale-knn-part-two/).

This notebook demonstrates how to use the Mixedbread [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) model with support for binary vectors with Vespa. The notebook example also includes a re-ranking phase that uses the float query vector version for improved accuracy. The re-ranking step makes the model perform at 96.45% of the full float version, with a 32x decrease in storage footprint.

Install the dependencies:

In \[ \]:

Copied!

```
!pip3 install -U pyvespa sentence-transformers vespacli
```

!pip3 install -U pyvespa sentence-transformers vespacli

## Examining the embeddings using sentence-transformers[¶](#examining-the-embeddings-using-sentence-transformers)

Read the [blog post](https://huggingface.co/blog/embedding-quantization) for `sentence-transformer` usage.

[sentence-transformer API](https://sbert.net/docs/package_reference/SentenceTransformer.html). Model card: [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).

Load the model using the sentence-transformers library:

In \[1\]:

Copied!

```
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1",
    prompts={
        "retrieval": "Represent this sentence for searching relevant passages: ",
    },
    default_prompt_name="retrieval",
)
```

from sentence_transformers import SentenceTransformer model = SentenceTransformer( "mixedbread-ai/mxbai-embed-large-v1", prompts={ "retrieval": "Represent this sentence for searching relevant passages: ", }, default_prompt_name="retrieval", )

```
Default prompt name is set to 'retrieval'. This prompt will be applied to all `encode()` calls, except if `encode()` is called with `prompt` or `prompt_name` parameters.
```

### Some sample documents[¶](#some-sample-documents)

Define a few sample documents that we want to embed

In \[4\]:

Copied!

```
documents = [
    "Alan Turing  was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist.",
    "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.",
    "Isaac Newton was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher.",
    "Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity",
]
```

documents = [ "Alan Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist.", "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.", "Isaac Newton was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher.", "Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity", ]

Run embedding inference, notice how we specify `precision="binary"`.

In \[5\]:

Copied!

```
binary_embeddings = model.encode(documents, precision="binary")
```

binary_embeddings = model.encode(documents, precision="binary")

In \[8\]:

Copied!

```
print(
    "Binary embedding shape {} with type {}".format(
        binary_embeddings.shape, binary_embeddings.dtype
    )
)
```

print( "Binary embedding shape {} with type {}".format( binary_embeddings.shape, binary_embeddings.dtype ) )

```
Binary embedding shape (4, 128) with type int8
```

## Defining the Vespa application[¶](#defining-the-vespa-application)

First, we define a [Vespa schema](https://docs.vespa.ai/en/schemas.html) with the fields we want to store and their type.

Notice the `binary_vector` field that defines an indexed (dense) Vespa tensor with the dimension name `x[128]`.

The indexing statement includes `index` which means that Vespa will use HNSW indexing for this field.

Also notice the configuration of [distance-metric](https://docs.vespa.ai/en/reference/schema-reference.html#distance-metric) where we specify `hamming`.

In \[9\]:

Copied!

```
from vespa.package import Schema, Document, Field, FieldSet

my_schema = Schema(
    name="doc",
    mode="index",
    document=Document(
        fields=[
            Field(
                name="doc_id",
                type="string",
                indexing=["summary", "index"],
                match=["word"],
                rank="filter",
            ),
            Field(
                name="text",
                type="string",
                indexing=["summary", "index"],
                index="enable-bm25",
            ),
            Field(
                name="binary_vector",
                type="tensor<int8>(x[128])",
                indexing=["attribute", "index"],
                attribute=["distance-metric: hamming"],
            ),
        ]
    ),
    fieldsets=[FieldSet(name="default", fields=["text"])],
)
```

from vespa.package import Schema, Document, Field, FieldSet my_schema = Schema( name="doc", mode="index", document=Document( fields=\[ Field( name="doc_id", type="string", indexing=["summary", "index"], match=["word"], rank="filter", ), Field( name="text", type="string", indexing=["summary", "index"], index="enable-bm25", ), Field( name="binary_vector", type="tensor<int8>(x[128])", indexing=["attribute", "index"], attribute=["distance-metric: hamming"], ), \] ), fieldsets=\[FieldSet(name="default", fields=["text"])\], )

We must add the schema to a Vespa [application package](https://docs.vespa.ai/en/application-packages.html). This consists of configuration files, schemas, models, and possibly even custom code (plugins).

In \[15\]:

Copied!

```
from vespa.package import ApplicationPackage

vespa_app_name = "mixedbreadai"
vespa_application_package = ApplicationPackage(name=vespa_app_name, schema=[my_schema])
```

from vespa.package import ApplicationPackage vespa_app_name = "mixedbreadai" vespa_application_package = ApplicationPackage(name=vespa_app_name, schema=[my_schema])

In the last step, we configure [ranking](https://docs.vespa.ai/en/ranking.html) by adding `rank-profile`'s to the schema.

`unpack_bits` unpacks the binary representation into a 1024-dimensional float vector [doc](https://docs.vespa.ai/en/reference/ranking-expressions.html#unpack-bits).

We define two tensor inputs, one compact binary representation that is used for the nearestNeighbor search and one full version that is used in ranking.

In \[16\]:

Copied!

```
from vespa.package import RankProfile, FirstPhaseRanking, SecondPhaseRanking, Function


rerank = RankProfile(
    name="rerank",
    inputs=[
        ("query(q_binary)", "tensor<int8>(x[128])"),
        ("query(q_full)", "tensor<float>(x[1024])"),
    ],
    functions=[
        Function(  # this returns a tensor<float>(x[1024]) with values -1 or 1
            name="unpack_binary_representation",
            expression="2*unpack_bits(attribute(binary_vector)) -1",
        )
    ],
    first_phase=FirstPhaseRanking(
        expression="closeness(field, binary_vector)"  # 1/(1 + hamming_distance). Calculated between the binary query and the binary_vector
    ),
    second_phase=SecondPhaseRanking(
        expression="sum( query(q_full)* unpack_binary_representation )",  # re-rank using the dot product between float query and the unpacked binary representation
        rerank_count=100,
    ),
    match_features=["distance(field, binary_vector)"],
)
my_schema.add_rank_profile(rerank)
```

from vespa.package import RankProfile, FirstPhaseRanking, SecondPhaseRanking, Function rerank = RankProfile( name="rerank", inputs=\[ ("query(q_binary)", "tensor<int8>(x[128])"), ("query(q_full)", "tensor<float>(x[1024])"), \], functions=\[ Function( # this returns a tensor<float>(x[1024]) with values -1 or 1 name="unpack_binary_representation", expression="2\*unpack_bits(attribute(binary_vector)) -1", ) \], first_phase=FirstPhaseRanking( expression="closeness(field, binary_vector)" # 1/(1 + hamming_distance). Calculated between the binary query and the binary_vector ), second_phase=SecondPhaseRanking( expression="sum( query(q_full)\* unpack_binary_representation )", # re-rank using the dot product between float query and the unpacked binary representation rerank_count=100, ), match_features=["distance(field, binary_vector)"], ) my_schema.add_rank_profile(rerank)

## Deploy the application to Vespa Cloud[¶](#deploy-the-application-to-vespa-cloud)

With the configured application, we can deploy it to [Vespa Cloud](https://cloud.vespa.ai/en/).

To deploy the application to Vespa Cloud we need to create a tenant in the Vespa Cloud:

Create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/) (unless you already have one). This step requires a Google or GitHub account, and will start your [free trial](https://cloud.vespa.ai/en/free-trial).

Make note of the tenant name, it is used in the next steps.

> Note: Deployments to dev and perf expire after 7 days of inactivity, i.e., 7 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 7 more days.

In \[22\]:

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

In \[23\]:

Copied!

```
from vespa.application import Vespa

app: Vespa = vespa_cloud.deploy()
```

from vespa.application import Vespa app: Vespa = vespa_cloud.deploy()

```
Deployment started in run 1 of dev-aws-us-east-1c for samples.mixedbreadai. This may take a few minutes the first time.
INFO    [22:14:39]  Deploying platform version 8.322.22 and application dev build 1 for dev-aws-us-east-1c of default ...
INFO    [22:14:39]  Using CA signed certificate version 0
INFO    [22:14:46]  Using 1 nodes in container cluster 'mixedbreadai_container'
INFO    [22:15:18]  Session 2205 for tenant 'samples' prepared and activated.
INFO    [22:15:21]  ######## Details for all nodes ########
INFO    [22:15:35]  h90193a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [22:15:35]  --- platform vespa/cloud-tenant-rhel8:8.322.22 <-- :
INFO    [22:15:35]  --- logserver-container on port 4080 has not started 
INFO    [22:15:35]  --- metricsproxy-container on port 19092 has not started 
INFO    [22:15:35]  h90971b.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [22:15:35]  --- platform vespa/cloud-tenant-rhel8:8.322.22 <-- :
INFO    [22:15:35]  --- container-clustercontroller on port 19050 has not started 
INFO    [22:15:35]  --- metricsproxy-container on port 19092 has not started 
INFO    [22:15:35]  h91168a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [22:15:35]  --- platform vespa/cloud-tenant-rhel8:8.322.22 <-- :
INFO    [22:15:35]  --- storagenode on port 19102 has not started 
INFO    [22:15:35]  --- searchnode on port 19107 has not started 
INFO    [22:15:35]  --- distributor on port 19111 has not started 
INFO    [22:15:35]  --- metricsproxy-container on port 19092 has not started 
INFO    [22:15:35]  h91567a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
INFO    [22:15:35]  --- platform vespa/cloud-tenant-rhel8:8.322.22 <-- :
INFO    [22:15:35]  --- container on port 4080 has not started 
INFO    [22:15:35]  --- metricsproxy-container on port 19092 has not started 
INFO    [22:16:41]  Waiting for convergence of 10 services across 4 nodes
INFO    [22:16:41]  1/1 nodes upgrading platform
INFO    [22:16:41]  2 application services still deploying
DEBUG   [22:16:41]  h91567a.dev.aws-us-east-1c.vespa-external.aws.oath.cloud: expected to be UP
DEBUG   [22:16:41]  --- platform vespa/cloud-tenant-rhel8:8.322.22 <-- :
DEBUG   [22:16:41]  --- container on port 4080 has not started 
DEBUG   [22:16:41]  --- metricsproxy-container on port 19092 has not started 
INFO    [22:17:11]  Found endpoints:
INFO    [22:17:11]  - dev.aws-us-east-1c
INFO    [22:17:11]   |-- https://cf949f23.b8a7f611.z.vespa-app.cloud/ (cluster 'mixedbreadai_container')
INFO    [22:17:12]  Installation succeeded!
Using mTLS (key,cert) Authentication against endpoint https://cf949f23.b8a7f611.z.vespa-app.cloud//ApplicationStatus
Application is up!
Finished deployment.
```

## Feed our sample documents and their binary embedding representation[¶](#feed-our-sample-documents-and-their-binary-embedding-representation)

With few documents, we use the synchronous API. Read more in [reads and writes](https://vespa-engine.github.io/pyvespa/reads-writes.md).

In \[24\]:

Copied!

```
from vespa.io import VespaResponse

for i, doc in enumerate(documents):
    response: VespaResponse = app.feed_data_point(
        schema="doc",
        data_id=str(i),
        fields={
            "doc_id": str(i),
            "text": doc,
            "binary_vector": binary_embeddings[i].tolist(),
        },
    )
    assert response.is_successful()
```

from vespa.io import VespaResponse for i, doc in enumerate(documents): response: VespaResponse = app.feed_data_point( schema="doc", data_id=str(i), fields={ "doc_id": str(i), "text": doc, "binary_vector": binary_embeddings[i].tolist(), }, ) assert response.is_successful()

### Querying data[¶](#querying-data)

Read more about querying Vespa in:

- [Vespa Query API](https://docs.vespa.ai/en/query-api.html)
- [Vespa Query API reference](https://docs.vespa.ai/en/reference/query-api-reference.html)
- [Vespa Query Language API (YQL)](https://docs.vespa.ai/en/query-language.html)
- [Practical Nearest Neighbor Search Guide](https://docs.vespa.ai/en/nearest-neighbor-search-guide.html)

In this case, we use [quantization.quantize_embeddings](https://sbert.net/docs/package_reference/quantization.html#sentence_transformers.quantization.quantize_embeddings) after first obtaining the float version, this to avoid running the model inference twice.

In \[54\]:

Copied!

```
query = "Who was Isac Newton?"
# This returns the float version
query_embedding_float = model.encode([query])
```

query = "Who was Isac Newton?"

# This returns the float version

query_embedding_float = model.encode([query])

In \[ \]:

Copied!

```
from sentence_transformers.quantization import quantize_embeddings

query_embedding_binary = quantize_embeddings(query_embedding_float, precision="binary")
```

from sentence_transformers.quantization import quantize_embeddings query_embedding_binary = quantize_embeddings(query_embedding_float, precision="binary")

Now, we use nearestNeighbor search to retrieve 100 hits (`targetHits`) using the configured distance-metric (hamming distance). The retrieved hits are exposed to the ‹espa ranking framework, where we re-rank using the dot product between the float tensor and the unpacked binary vector.

In \[55\]:

Copied!

```
response = app.query(
    yql="select * from doc where {targetHits:100}nearestNeighbor(binary_vector,q_binary)",
    ranking="rerank",
    body={
        "input.query(q_binary)": query_embedding_binary[0].tolist(),
        "input.query(q_full)": query_embedding_float[0].tolist(),
    },
)
assert response.is_successful()
```

response = app.query( yql="select * from doc where {targetHits:100}nearestNeighbor(binary_vector,q_binary)", ranking="rerank", body={ "input.query(q_binary)": query_embedding_binary[0].tolist(), "input.query(q_full)": query_embedding_float[0].tolist(), }, ) assert response.is_successful()

In \[56\]:

Copied!

```
import json

print(json.dumps(response.hits, indent=2))
```

import json print(json.dumps(response.hits, indent=2))

```
[
  {
    "id": "id:doc:doc::2",
    "relevance": 177.8957977294922,
    "source": "mixedbreadai_content",
    "fields": {
      "matchfeatures": {
        "closeness(field,binary_vector)": 0.003484320557491289,
        "distance(field,binary_vector)": 286.0
      },
      "sddocname": "doc",
      "documentid": "id:doc:doc::2",
      "doc_id": "2",
      "text": "Isaac Newton was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher."
    }
  },
  {
    "id": "id:doc:doc::1",
    "relevance": 144.52731323242188,
    "source": "mixedbreadai_content",
    "fields": {
      "matchfeatures": {
        "closeness(field,binary_vector)": 0.002890173410404624,
        "distance(field,binary_vector)": 345.0
      },
      "sddocname": "doc",
      "documentid": "id:doc:doc::1",
      "doc_id": "1",
      "text": "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time."
    }
  },
  {
    "id": "id:doc:doc::0",
    "relevance": 138.78799438476562,
    "source": "mixedbreadai_content",
    "fields": {
      "matchfeatures": {
        "closeness(field,binary_vector)": 0.00273224043715847,
        "distance(field,binary_vector)": 365.0
      },
      "sddocname": "doc",
      "documentid": "id:doc:doc::0",
      "doc_id": "0",
      "text": "Alan Turing  was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist."
    }
  },
  {
    "id": "id:doc:doc::3",
    "relevance": 115.2405776977539,
    "source": "mixedbreadai_content",
    "fields": {
      "matchfeatures": {
        "closeness(field,binary_vector)": 0.002652519893899204,
        "distance(field,binary_vector)": 376.0
      },
      "sddocname": "doc",
      "documentid": "id:doc:doc::3",
      "doc_id": "3",
      "text": "Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity"
    }
  }
]
```

## Summary[¶](#summary)

Binary embeddings is an exciting development, as it reduces storage (32) and speed up vector searches as the hamming distance is much more efficient than distance metrics like angular or euclidean.

### Clean up[¶](#clean-up)

We can now delete the cloud instance:

In \[ \]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()
