# Standalone ColBERT + Vespa for long-context ranking[¶](#standalone-colbert-vespa-for-long-context-ranking)

This is a guide on how to use the [ColBERT](https://github.com/stanford-futuredata/ColBERT) package to produce token-level vectors. This as an alternative for using the native Vespa [colbert embedder](https://docs.vespa.ai/en/embedding.html#colbert-embedder).

This guide illustrates how to feed multiple passages per Vespa document (long-context)

- Compress token vectors using binarization compatible with Vespa `unpack_bits`
- Use Vespa hex feed format for binary vectors with mixed vespa tensors
- How to query Vespa with the ColBERT query tensor representation

Read more about [Vespa Long-Context ColBERT](https://blog.vespa.ai/announcing-long-context-colbert-in-vespa/).

In \[ \]:

Copied!

```
!pip3 install -U pyvespa colbert-ai numpy torch vespacli transformers<=4.49.0
```

!pip3 install -U pyvespa colbert-ai numpy torch vespacli transformers\<=4.49.0

Load a checkpoint with ColBERT and obtain document and query embeddings

In \[ \]:

Copied!

```
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig

ckpt = Checkpoint(
    "colbert-ir/colbertv2.0", colbert_config=ColBERTConfig(root="experiments")
)
```

from colbert.modeling.checkpoint import Checkpoint from colbert.infra import ColBERTConfig ckpt = Checkpoint( "colbert-ir/colbertv2.0", colbert_config=ColBERTConfig(root="experiments") )

A few sample documents:

In \[50\]:

Copied!

```
document_passages = [
    "Alan Turing  was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist.",
    "Born in Maida Vale, London, Turing was raised in southern England. He graduated from King's College, Cambridge, with a degree in mathematics.",
    "After the war, Turing worked at the National Physical Laboratory, where he designed the Automatic Computing Engine, one of the first designs for a stored-program computer.",
    "Turing has an extensive legacy with statues of him and many things named after him, including an annual award for computer science innovations.",
]
```

document_passages = [ "Alan Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist.", "Born in Maida Vale, London, Turing was raised in southern England. He graduated from King's College, Cambridge, with a degree in mathematics.", "After the war, Turing worked at the National Physical Laboratory, where he designed the Automatic Computing Engine, one of the first designs for a stored-program computer.", "Turing has an extensive legacy with statues of him and many things named after him, including an annual award for computer science innovations.", ]

In \[ \]:

Copied!

```
document_token_vectors = ckpt.docFromText(document_passages)
```

document_token_vectors = ckpt.docFromText(document_passages)

See the shape of the ColBERT document embeddings:

In \[52\]:

Copied!

```
document_token_vectors.shape
```

document_token_vectors.shape

Out\[52\]:

```
torch.Size([4, 35, 128])
```

In \[53\]:

Copied!

```
query_vectors = ckpt.queryFromText(["Who was Alan Turing?"])[0]
query_vectors.shape
```

query_vectors = ckpt.queryFromText(["Who was Alan Turing?"])[0] query_vectors.shape

Out\[53\]:

```
torch.Size([32, 128])
```

The query is always padded to 32 so in the above we have 32 query token vectors.

Routines for binarization and output in Vespa tensor format that can be used in queries and JSON feed.

In \[67\]:

Copied!

```
import numpy as np
import torch
from binascii import hexlify
from typing import List, Dict


def binarize_token_vectors_hex(vectors: torch.Tensor) -> Dict[str, str]:
    # Notice axix=2 to pack the bits in the last dimension, which is the token level vectors
    binarized_token_vectors = np.packbits(np.where(vectors > 0, 1, 0), axis=2).astype(
        np.int8
    )
    vespa_tensor = list()
    for chunk_index in range(0, len(binarized_token_vectors)):
        token_vectors = binarized_token_vectors[chunk_index]
        for token_index in range(0, len(token_vectors)):
            values = str(hexlify(token_vectors[token_index].tobytes()), "utf-8")
            if (
                values == "00000000000000000000000000000000"
            ):  # skip empty vectors due to padding with batch of passages
                continue
            vespa_tensor_cell = {
                "address": {"context": chunk_index, "token": token_index},
                "values": values,
            }
            vespa_tensor.append(vespa_tensor_cell)

    return vespa_tensor


def float_query_token_vectors(vectors: torch.Tensor) -> Dict[str, List[float]]:
    vespa_token_feed = dict()
    for index in range(0, len(vectors)):
        vespa_token_feed[index] = vectors[index].tolist()
    return vespa_token_feed
```

import numpy as np import torch from binascii import hexlify from typing import List, Dict def binarize_token_vectors_hex(vectors: torch.Tensor) -> Dict\[str, str\]:

# Notice axix=2 to pack the bits in the last dimension, which is the token level vectors

binarized_token_vectors = np.packbits(np.where(vectors > 0, 1, 0), axis=2).astype( np.int8 ) vespa_tensor = list() for chunk_index in range(0, len(binarized_token_vectors)): token_vectors = binarized_token_vectors[chunk_index] for token_index in range(0, len(token_vectors)): values = str(hexlify(token_vectors[token_index].tobytes()), "utf-8") if ( values == "00000000000000000000000000000000" ): # skip empty vectors due to padding with batch of passages continue vespa_tensor_cell = { "address": {"context": chunk_index, "token": token_index}, "values": values, } vespa_tensor.append(vespa_tensor_cell) return vespa_tensor def float_query_token_vectors(vectors: torch.Tensor) -> Dict\[str, List[float]\]: vespa_token_feed = dict() for index in range(0, len(vectors)): vespa_token_feed[index] = vectors[index].tolist() return vespa_token_feed

In \[ \]:

Copied!

```
import json

print(json.dumps(binarize_token_vectors_hex(document_token_vectors)))
print(json.dumps(float_query_token_vectors(query_vectors)))
```

import json print(json.dumps(binarize_token_vectors_hex(document_token_vectors))) print(json.dumps(float_query_token_vectors(query_vectors)))

## Defining the Vespa application[¶](#defining-the-vespa-application)

[PyVespa](https://vespa-engine.github.io/pyvespa/) helps us build the [Vespa application package](https://docs.vespa.ai/en/application-packages.html). A Vespa application package consists of configuration files, schemas, models, and code (plugins).

First, we define a [Vespa schema](https://docs.vespa.ai/en/schemas.html) with the fields we want to store and their type.

In \[60\]:

Copied!

```
from vespa.package import Schema, Document, Field

colbert_schema = Schema(
    name="doc",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary"]),
            Field(
                name="passages",
                type="array<string>",
                indexing=["summary", "index"],
                index="enable-bm25",
            ),
            Field(
                name="colbert",
                type="tensor<int8>(context{}, token{}, v[16])",
                indexing=["attribute", "summary"],
            ),
        ]
    ),
)
```

from vespa.package import Schema, Document, Field colbert_schema = Schema( name="doc", document=Document( fields=\[ Field(name="id", type="string", indexing=["summary"]), Field( name="passages", type="array<string>", indexing=["summary", "index"], index="enable-bm25", ), Field( name="colbert", type="tensor<int8>(context{}, token{}, v[16])", indexing=["attribute", "summary"], ), \] ), )

In \[61\]:

Copied!

```
from vespa.package import ApplicationPackage

vespa_app_name = "colbertlong"
vespa_application_package = ApplicationPackage(
    name=vespa_app_name, schema=[colbert_schema]
)
```

from vespa.package import ApplicationPackage vespa_app_name = "colbertlong" vespa_application_package = ApplicationPackage( name=vespa_app_name, schema=[colbert_schema] )

Note that we use max sim in the first phase ranking over all the hits that are retrieved by the query logic. Also note that asymmetric MaxSim where we use `unpack_bits` to obtain a 128-d float vector representation from the binary vector representation.

In \[62\]:

Copied!

```
from vespa.package import RankProfile, Function, FirstPhaseRanking

colbert_profile = RankProfile(
    name="default",
    inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
    functions=[
        Function(
            name="max_sim_per_context",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(colbert)) , v
                        ),
                        max, token
                    ),
                    querytoken
                )
            """,
        ),
        Function(
            name="max_sim", expression="reduce(max_sim_per_context, max, context)"
        ),
    ],
    first_phase=FirstPhaseRanking(expression="max_sim"),
    match_features=["max_sim_per_context"],
)
colbert_schema.add_rank_profile(colbert_profile)
```

from vespa.package import RankProfile, Function, FirstPhaseRanking colbert_profile = RankProfile( name="default", inputs=\[("query(qt)", "tensor<float>(querytoken{}, v[128])")\], functions=[ Function( name="max_sim_per_context", expression=""" sum( reduce( sum( query(qt) * unpack_bits(attribute(colbert)) , v ), max, token ), querytoken ) """, ), Function( name="max_sim", expression="reduce(max_sim_per_context, max, context)" ), ], first_phase=FirstPhaseRanking(expression="max_sim"), match_features=["max_sim_per_context"], ) colbert_schema.add_rank_profile(colbert_profile)

## Deploy the application to Vespa Cloud[¶](#deploy-the-application-to-vespa-cloud)

With the configured application, we can deploy it to [Vespa Cloud](https://cloud.vespa.ai/en/).

To deploy the application to Vespa Cloud we need to create a tenant in the Vespa Cloud:

Create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/) (unless you already have one). This step requires a Google or GitHub account, and will start your [free trial](https://cloud.vespa.ai/en/free-trial).

Make note of the tenant name, it is used in the next steps.

> Note: Deployments to dev and perf expire after 7 days of inactivity, i.e., 7 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 7 more days.

In \[63\]:

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

In \[ \]:

Copied!

```
from vespa.application import Vespa

app: Vespa = vespa_cloud.deploy()
```

from vespa.application import Vespa app: Vespa = vespa_cloud.deploy()

Use Vespa tensor `blocks` format for mixed tensors (two mapped dimensions with one dense) [doc](https://docs.vespa.ai/en/reference/document-json-format.html#tensor).

In \[65\]:

Copied!

```
from vespa.io import VespaResponse

vespa_feed_format = {
    "id": "1",
    "passages": document_passages,
    "colbert": {"blocks": binarize_token_vectors_hex(document_token_vectors)},
}
# synchrounous feed (this is blocking and slow, but few docs..)
with app.syncio() as sync:
    response: VespaResponse = sync.feed_data_point(
        data_id=1, fields=vespa_feed_format, schema="doc"
    )
```

from vespa.io import VespaResponse vespa_feed_format = { "id": "1", "passages": document_passages, "colbert": {"blocks": binarize_token_vectors_hex(document_token_vectors)}, }

# synchrounous feed (this is blocking and slow, but few docs..)

with app.syncio() as sync: response: VespaResponse = sync.feed_data_point( data_id=1, fields=vespa_feed_format, schema="doc" )

### Querying Vespa with ColBERT tensors[¶](#querying-vespa-with-colbert-tensors)

This example uses brute-force "true" search without a retrieval step using nearestNeighbor or keywords.

In \[ \]:

Copied!

```
from vespa.io import VespaQueryResponse
import json

response: VespaQueryResponse = app.query(
    yql="select * from doc where true",
    ranking="default",
    body={
        "presentation.format.tensors": "short-value",
        "input.query(qt)": float_query_token_vectors(query_vectors),
    },
)
assert response.is_successful()
```

from vespa.io import VespaQueryResponse import json response: VespaQueryResponse = app.query( yql="select * from doc where true", ranking="default", body={ "presentation.format.tensors": "short-value", "input.query(qt)": float_query_token_vectors(query_vectors), }, ) assert response.is_successful()

You should see output similar to this:

```
{
  "id": "id:doc:doc::1",
  "relevance": 100.0651626586914,
  "source": "colbertlong_content",
  "fields": {
    "matchfeatures": {
      "max_sim_per_context": {
        "0": 100.0651626586914,
        "1": 62.7861328125,
        "2": 67.44772338867188,
        "3": 60.133323669433594
      }
    },
    "sddocname": "doc",
    "documentid": "id:doc:doc::1",
    "id": "1",
    "passages": [
      "Alan Turing  was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist.",
      "Born in Maida Vale, London, Turing was raised in southern England. He graduated from King's College, Cambridge, with a degree in mathematics.",
      "After the war, Turing worked at the National Physical Laboratory, where he designed the Automatic Computing Engine, one of the first designs for a stored-program computer.",
      "Turing has an extensive legacy with statues of him and many things named after him, including an annual award for computer science innovations."
    ],
    "colbert": [
      {
        "address": {
          "context": "0",
          "token": "0"
        },
        "values": [
          1,
          120,
          69,
          0,
          33,
          -60,
          -58,
          -95,
          -120,
          32,
          -127,
          67,
          -51,
          68,
          -106,
          -12
        ]
      },
      {
        "address": {
          "context": "0",
          "token": "1"
        },
        "values": [
          -122,
          60,
          9,
          -128,
          97,
          -60,
          -58,
          -95,
          -80,
          112,
          -127,
          67,
          -99,
          68,
          -106,
          -28
        ]
      },
      "..."
    ],

  }
}
```

As can be seen from the matchfeatures, the first context (index 0) scored the highest and this is the score that is used to score the entire document.

In \[ \]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()
