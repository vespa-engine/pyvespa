# Standalone ColBERT with Vespa for end-to-end retrieval and ranking[¶](#standalone-colbert-with-vespa-for-end-to-end-retrieval-and-ranking)

This notebook illustrates using [ColBERT](https://github.com/stanford-futuredata/ColBERT) package to produce token vectors, instead of using the native Vespa [colbert embedder](https://docs.vespa.ai/en/embedding.html#colbert-embedder).

This guide illustrates how to feed and query using a single passage representation

- Compress token vectors using binarization compatible with Vespa unpackbits used in ranking. This implements the binarization of token-level vectors using `numpy`.
- Use Vespa hex feed format for binary vectors [doc](https://docs.vespa.ai/en/reference/document-json-format.html#tensor).
- Query examples.

As a bonus, this also demonstrates how to use ColBERT end-to-end with Vespa for both retrieval and ranking. The retrieval step searches the binary token-level representations using hamming distance. This uses 32 nearestNeighbor operators in the same query, each finding 100 nearest hits in hamming space. Then the results are re-ranked using the full-blown MaxSim calculation.

See [Announcing the Vespa ColBERT embedder](https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/) for details on ColBERT and the binary quantization used to compress ColBERT's token-level vectors.

In \[ \]:

Copied!

```
!pip3 install -U pyvespa colbert-ai numpy torch transformers<=4.49.0
```

!pip3 install -U pyvespa colbert-ai numpy torch transformers\<=4.49.0

Load a checkpoint with colbert and obtain document and query embeddings

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

In \[139\]:

Copied!

```
passage = [
    "Alan Mathison Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist."
]
```

passage = [ "Alan Mathison Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist." ]

In \[ \]:

Copied!

```
vectors = ckpt.docFromText(passage)[0]
```

vectors = ckpt.docFromText(passage)[0]

In \[129\]:

Copied!

```
vectors.shape
```

vectors.shape

Out\[129\]:

```
torch.Size([27, 128])
```

In this case, we got 27 token-level embeddings, each using 128 float dimensions. This includes CLS token and special tokens used to differentiate the query from the document encoding.

In \[130\]:

Copied!

```
query_vectors = ckpt.queryFromText(["Who was Alan Turing?"])[0]
query_vectors.shape
```

query_vectors = ckpt.queryFromText(["Who was Alan Turing?"])[0] query_vectors.shape

Out\[130\]:

```
torch.Size([32, 128])
```

Routines for binarization and output in Vespa tensor format that can be used in queries and in JSON feed.

In \[118\]:

Copied!

```
import numpy as np
import torch
from binascii import hexlify
from typing import Dict, List


def binarize_token_vectors_hex(vectors: torch.Tensor) -> Dict[str, str]:
    binarized_token_vectors = np.packbits(np.where(vectors > 0, 1, 0), axis=1).astype(
        np.int8
    )
    vespa_token_feed = dict()
    for index in range(0, len(binarized_token_vectors)):
        vespa_token_feed[index] = str(
            hexlify(binarized_token_vectors[index].tobytes()), "utf-8"
        )
    return vespa_token_feed


def float_query_token_vectors(vectors: torch.Tensor) -> Dict[str, List[float]]:
    vespa_token_feed = dict()
    for index in range(0, len(vectors)):
        vespa_token_feed[index] = vectors[index].tolist()
    return vespa_token_feed
```

import numpy as np import torch from binascii import hexlify from typing import Dict, List def binarize_token_vectors_hex(vectors: torch.Tensor) -> Dict\[str, str\]: binarized_token_vectors = np.packbits(np.where(vectors > 0, 1, 0), axis=1).astype( np.int8 ) vespa_token_feed = dict() for index in range(0, len(binarized_token_vectors)): vespa_token_feed[index] = str( hexlify(binarized_token_vectors[index].tobytes()), "utf-8" ) return vespa_token_feed def float_query_token_vectors(vectors: torch.Tensor) -> Dict\[str, List[float]\]: vespa_token_feed = dict() for index in range(0, len(vectors)): vespa_token_feed[index] = vectors[index].tolist() return vespa_token_feed

In \[ \]:

Copied!

```
import json

print(json.dumps(binarize_token_vectors_hex(vectors)))
print(json.dumps(float_query_token_vectors(query_vectors)))
```

import json print(json.dumps(binarize_token_vectors_hex(vectors))) print(json.dumps(float_query_token_vectors(query_vectors)))

## Defining the Vespa application[¶](#defining-the-vespa-application)

[PyVespa](https://vespa-engine.github.io/pyvespa/) helps us build the [Vespa application package](https://docs.vespa.ai/en/application-packages.html). A Vespa application package consists of configuration files, schemas, models, and code (plugins).

First, we define a [Vespa schema](https://docs.vespa.ai/en/schemas.html) with the fields we want to store and their type.

We use HNSW with hamming distance for retrieval

In \[151\]:

Copied!

```
from vespa.package import Schema, Document, Field

colbert_schema = Schema(
    name="doc",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary"]),
            Field(name="passage", type="string", indexing=["index", "summary"]),
            Field(
                name="colbert",
                type="tensor<int8>(token{}, v[16])",
                indexing=["attribute", "summary", "index"],
                attribute=["distance-metric:hamming"],
            ),
        ]
    ),
)
```

from vespa.package import Schema, Document, Field colbert_schema = Schema( name="doc", document=Document( fields=\[ Field(name="id", type="string", indexing=["summary"]), Field(name="passage", type="string", indexing=["index", "summary"]), Field( name="colbert", type="tensor<int8>(token{}, v[16])", indexing=["attribute", "summary", "index"], attribute=["distance-metric:hamming"], ), \] ), )

In \[152\]:

Copied!

```
from vespa.package import ApplicationPackage

vespa_app_name = "colbert"
vespa_application_package = ApplicationPackage(
    name=vespa_app_name, schema=[colbert_schema]
)
```

from vespa.package import ApplicationPackage vespa_app_name = "colbert" vespa_application_package = ApplicationPackage( name=vespa_app_name, schema=[colbert_schema] )

We need to define all the query input tensors. We are going to input up to 32 query tensors in binary form these are used for retrieval

In \[92\]:

Copied!

```
query_binary_input_tensors = []
for index in range(0, 32):
    query_binary_input_tensors.append(
        ("query(binary_vector_{})".format(index), "tensor<int8>(v[16])")
    )
```

query_binary_input_tensors = [] for index in range(0, 32): query_binary_input_tensors.append( ("query(binary_vector\_{})".format(index), "tensor<int8>(v[16])") )

Note that we just use max sim in the first phase ranking over all the hits that are retrieved by the query

In \[153\]:

Copied!

```
from vespa.package import RankProfile, Function, FirstPhaseRanking

colbert = RankProfile(
    name="default",
    inputs=[
        ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
        *query_binary_input_tensors,
    ],
    functions=[
        Function(
            name="max_sim",
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
        )
    ],
    first_phase=FirstPhaseRanking(expression="max_sim"),
)
colbert_schema.add_rank_profile(colbert)
```

from vespa.package import RankProfile, Function, FirstPhaseRanking colbert = RankProfile( name="default", inputs=\[ ("query(qt)", "tensor<float>(querytoken{}, v[128])"), \*query_binary_input_tensors, \], functions=[ Function( name="max_sim", expression=""" sum( reduce( sum( query(qt) * unpack_bits(attribute(colbert)) , v ), max, token ), querytoken ) """, ) ], first_phase=FirstPhaseRanking(expression="max_sim"), ) colbert_schema.add_rank_profile(colbert)

## Deploy the application to Vespa Cloud[¶](#deploy-the-application-to-vespa-cloud)

With the configured application, we can deploy it to [Vespa Cloud](https://cloud.vespa.ai/en/). It is also possible to deploy the app using docker; see the [Hybrid Search - Quickstart](https://vespa-engine.github.io/pyvespa/getting-started-pyvespa.md) guide for an example of deploying it to a local docker container.

Install the Vespa CLI.

In \[ \]:

Copied!

```
!pip3 install vespacli
```

!pip3 install vespacli

To deploy the application to Vespa Cloud we need to create a tenant in the Vespa Cloud:

Create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/) (unless you already have one). This step requires a Google or GitHub account, and will start your [free trial](https://cloud.vespa.ai/en/free-trial). Make note of the tenant name, it is used in the next steps.

### Configure Vespa Cloud date-plane security[¶](#configure-vespa-cloud-date-plane-security)

Create Vespa Cloud data-plane mTLS cert/key-pair. The mutual certificate pair is used to talk to your Vespa cloud endpoints. See [Vespa Cloud Security Guide](https://cloud.vespa.ai/en/security/guide) for details.

We save the paths to the credentials for later data-plane access without using pyvespa APIs.

In \[ \]:

Copied!

```
import os

os.environ["TENANT_NAME"] = "vespa-team"  # Replace with your tenant name

vespa_cli_command = (
    f'vespa config set application {os.environ["TENANT_NAME"]}.{vespa_app_name}'
)

!vespa config set target cloud
!{vespa_cli_command}
!vespa auth cert -N
```

import os os.environ["TENANT_NAME"] = "vespa-team" # Replace with your tenant name vespa_cli_command = ( f'vespa config set application {os.environ["TENANT_NAME"]}.{vespa_app_name}' ) !vespa config set target cloud !{vespa_cli_command} !vespa auth cert -N

Validate that we have the expected data-plane credential files:

In \[52\]:

Copied!

```
from os.path import exists
from pathlib import Path

cert_path = (
    Path.home()
    / ".vespa"
    / f"{os.environ['TENANT_NAME']}.{vespa_app_name}.default/data-plane-public-cert.pem"
)
key_path = (
    Path.home()
    / ".vespa"
    / f"{os.environ['TENANT_NAME']}.{vespa_app_name}.default/data-plane-private-key.pem"
)

if not exists(cert_path) or not exists(key_path):
    print(
        "ERROR: set the correct paths to security credentials. Correct paths above and rerun until you do not see this error"
    )
```

from os.path import exists from pathlib import Path cert_path = ( Path.home() / ".vespa" / f"{os.environ['TENANT_NAME']}.{vespa_app_name}.default/data-plane-public-cert.pem" ) key_path = ( Path.home() / ".vespa" / f"{os.environ['TENANT_NAME']}.{vespa_app_name}.default/data-plane-private-key.pem" ) if not exists(cert_path) or not exists(key_path): print( "ERROR: set the correct paths to security credentials. Correct paths above and rerun until you do not see this error" )

Note that the subsequent Vespa Cloud deploy call below will add `data-plane-public-cert.pem` to the application before deploying it to Vespa Cloud, so that you have access to both the private key and the public certificate. At the same time, Vespa Cloud only knows the public certificate.

### Configure Vespa Cloud control-plane security[¶](#configure-vespa-cloud-control-plane-security)

Authenticate to generate a tenant level control plane API key for deploying the applications to Vespa Cloud, and save the path to it.

The generated tenant api key must be added in the Vespa Console before attempting to deploy the application.

```
To use this key in Vespa Cloud click 'Add custom key' at
https://console.vespa-cloud.com/tenant/TENANT_NAME/account/keys
and paste the entire public key including the BEGIN and END lines.
```

In \[ \]:

Copied!

```
!vespa auth api-key

from pathlib import Path

api_key_path = Path.home() / ".vespa" / f"{os.environ['TENANT_NAME']}.api-key.pem"
```

!vespa auth api-key from pathlib import Path api_key_path = Path.home() / ".vespa" / f"{os.environ['TENANT_NAME']}.api-key.pem"

### Deploy to Vespa Cloud[¶](#deploy-to-vespa-cloud)

Now that we have data-plane and control-plane credentials ready, we can deploy our application to Vespa Cloud!

`PyVespa` supports deploying apps to the [development zone](https://cloud.vespa.ai/en/reference/environments#dev-and-perf).

> Note: Deployments to dev and perf expire after 7 days of inactivity, i.e., 7 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 7 more days.

In \[154\]:

Copied!

```
from vespa.deployment import VespaCloud


def read_secret():
    """Read the API key from the environment variable. This is
    only used for CI/CD purposes."""
    t = os.getenv("VESPA_TEAM_API_KEY")
    if t:
        return t.replace(r"\n", "\n")
    else:
        return t


vespa_cloud = VespaCloud(
    tenant=os.environ["TENANT_NAME"],
    application=vespa_app_name,
    key_content=read_secret() if read_secret() else None,
    key_location=api_key_path,
    application_package=vespa_application_package,
)
```

from vespa.deployment import VespaCloud def read_secret(): """Read the API key from the environment variable. This is only used for CI/CD purposes.""" t = os.getenv("VESPA_TEAM_API_KEY") if t: return t.replace(r"\\n", "\\n") else: return t vespa_cloud = VespaCloud( tenant=os.environ["TENANT_NAME"], application=vespa_app_name, key_content=read_secret() if read_secret() else None, key_location=api_key_path, application_package=vespa_application_package, )

Now deploy the app to Vespa Cloud dev zone.

The first deployment typically takes 2 minutes until the endpoint is up.

In \[ \]:

Copied!

```
from vespa.application import Vespa

app: Vespa = vespa_cloud.deploy()
```

from vespa.application import Vespa app: Vespa = vespa_cloud.deploy()

In \[156\]:

Copied!

```
from vespa.io import VespaResponse

vespa_feed_format = {
    "id": "1",
    "passage": passage[0],
    "colbert": binarize_token_vectors_hex(vectors),
}
with app.syncio() as sync:
    response: VespaResponse = sync.feed_data_point(
        data_id=1, fields=vespa_feed_format, schema="doc"
    )
```

from vespa.io import VespaResponse vespa_feed_format = { "id": "1", "passage": passage[0], "colbert": binarize_token_vectors_hex(vectors), } with app.syncio() as sync: response: VespaResponse = sync.feed_data_point( data_id=1, fields=vespa_feed_format, schema="doc" )

## Querying[¶](#querying)

Now we create all the query token vectors in binary form and use 32 nearestNeighbor query operators that are combined with OR. These hits are then exposed to ranking where the final MaxSim is performed using the unpacked binary representations.

In \[ \]:

Copied!

```
query_vectors = ckpt.queryFromText(["Who was Alan Turing?"])[0]
binary_query_input_tensors = binarize_token_vectors_hex(query_vectors)
```

query_vectors = ckpt.queryFromText(["Who was Alan Turing?"])[0] binary_query_input_tensors = binarize_token_vectors_hex(query_vectors)

In \[158\]:

Copied!

```
binary_query_vectors = dict()
nn_operators = list()
for index in range(0, 32):
    name = "input.query(binary_vector_{})".format(index)
    nn_argument = "binary_vector_{}".format(index)
    value = binary_query_input_tensors[index]
    binary_query_vectors[name] = value
    nn_operators.append("({targetHits:100}nearestNeighbor(colbert, %s))" % nn_argument)
```

binary_query_vectors = dict() nn_operators = list() for index in range(0, 32): name = "input.query(binary_vector\_{})".format(index) nn_argument = "binary_vector\_{}".format(index) value = binary_query_input_tensors[index] binary_query_vectors[name] = value nn_operators.append("({targetHits:100}nearestNeighbor(colbert, %s))" % nn_argument)

In \[159\]:

Copied!

```
nn_operators = " OR ".join(nn_operators)
```

nn_operators = " OR ".join(nn_operators)

Out\[159\]:

```
'({targetHits:100}nearestNeighbor(colbert, binary_vector_0)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_1)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_2)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_3)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_4)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_5)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_6)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_7)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_8)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_9)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_10)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_11)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_12)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_13)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_14)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_15)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_16)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_17)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_18)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_19)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_20)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_21)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_22)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_23)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_24)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_25)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_26)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_27)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_28)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_29)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_30)) OR ({targetHits:100}nearestNeighbor(colbert, binary_vector_31))'
```

In \[161\]:

Copied!

```
from vespa.io import VespaQueryResponse
import json

response: VespaQueryResponse = app.query(
    yql="select * from doc where {}".format(nn_operators),
    ranking="default",
    body={
        "presentation.format.tensors": "short-value",
        "input.query(qt)": float_query_token_vectors(query_vectors),
        **binary_query_vectors,
    },
)
assert response.is_successful()
print(json.dumps(response.hits[0], indent=2))
```

from vespa.io import VespaQueryResponse import json response: VespaQueryResponse = app.query( yql="select * from doc where {}".format(nn_operators), ranking="default", body={ "presentation.format.tensors": "short-value", "input.query(qt)": float_query_token_vectors(query_vectors), \*\*binary_query_vectors, }, ) assert response.is_successful() print(json.dumps(response.hits[0], indent=2))

```
{
  "id": "id:doc:doc::1",
  "relevance": 100.57648777961731,
  "source": "colbert_content",
  "fields": {
    "sddocname": "doc",
    "documentid": "id:doc:doc::1",
    "id": "1",
    "passage": "Alan Mathison Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist.",
    "colbert": {
      "0": [
        3,
        120,
        69,
        0,
        37,
        -60,
        -58,
        -95,
        -120,
        32,
        -127,
        67,
        -36,
        68,
        -106,
        -12
      ],
      "1": [
        -106,
        40,
        -119,
        -128,
        96,
        -60,
        -58,
        33,
        48,
        96,
        -127,
        67,
        -100,
        96,
        -106,
        -12
      ],
      "2": [
        -28,
        -84,
        73,
        -18,
        113,
        -60,
        -51,
        40,
        -96,
        121,
        4,
        24,
        -99,
        68,
        -47,
        -60
      ],
      "3": [
        -13,
        40,
        75,
        -124,
        65,
        64,
        -32,
        -53,
        12,
        64,
        125,
        4,
        24,
        -64,
        -69,
        101
      ],
      "4": [
        33,
        -54,
        113,
        24,
        77,
        -36,
        -44,
        3,
        -32,
        -72,
        40,
        41,
        -38,
        102,
        53,
        -35
      ],
      "5": [
        3,
        -22,
        73,
        -95,
        73,
        -51,
        85,
        -128,
        -121,
        25,
        17,
        68,
        90,
        64,
        -113,
        -28
      ],
      "6": [
        -109,
        -72,
        -114,
        0,
        97,
        -58,
        -57,
        -95,
        40,
        -96,
        -112,
        67,
        -97,
        -85,
        -42,
        -12
      ],
      "7": [
        -112,
        56,
        -114,
        0,
        97,
        -58,
        -57,
        -83,
        40,
        -96,
        -127,
        67,
        -97,
        43,
        -42,
        -12
      ],
      "8": [
        22,
        -71,
        65,
        96,
        0,
        -60,
        108,
        37,
        16,
        106,
        -55,
        115,
        -117,
        -56,
        -28,
        -12
      ],
      "9": [
        -106,
        -72,
        94,
        30,
        32,
        -60,
        -60,
        -19,
        24,
        -56,
        -47,
        -63,
        -40,
        -53,
        -103,
        -11
      ],
      "10": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "11": [
        -126,
        121,
        3,
        -103,
        32,
        70,
        103,
        -23,
        88,
        -55,
        -61,
        71,
        -101,
        -106,
        -8,
        -68
      ],
      "12": [
        18,
        24,
        -106,
        30,
        36,
        -42,
        -60,
        104,
        57,
        -120,
        -128,
        -61,
        -67,
        -53,
        -100,
        -11
      ],
      "13": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "14": [
        22,
        49,
        -38,
        17,
        36,
        -42,
        -25,
        65,
        25,
        -56,
        -45,
        -59,
        -102,
        -2,
        -65,
        125
      ],
      "15": [
        -105,
        25,
        -50,
        16,
        0,
        -42,
        -28,
        45,
        48,
        -56,
        -112,
        -55,
        -3,
        -87,
        -112,
        -11
      ],
      "16": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "17": [
        55,
        43,
        -62,
        33,
        -91,
        68,
        99,
        32,
        72,
        10,
        -41,
        70,
        -117,
        -78,
        -73,
        -11
      ],
      "18": [
        3,
        53,
        -117,
        20,
        36,
        -42,
        79,
        33,
        9,
        -120,
        -41,
        69,
        -36,
        -69,
        -111,
        117
      ],
      "19": [
        23,
        16,
        -42,
        20,
        44,
        -42,
        -26,
        33,
        57,
        -120,
        -112,
        -63,
        -3,
        -24,
        -108,
        -11
      ],
      "20": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "21": [
        -110,
        53,
        -106,
        28,
        32,
        -42,
        -58,
        77,
        61,
        -56,
        -42,
        -15,
        -68,
        -5,
        -110,
        -11
      ],
      "22": [
        -109,
        56,
        -114,
        0,
        96,
        -42,
        -58,
        -83,
        40,
        -96,
        -128,
        -61,
        -99,
        -21,
        -44,
        -12
      ],
      "23": [
        18,
        57,
        -50,
        30,
        36,
        86,
        -60,
        69,
        9,
        -120,
        -48,
        -63,
        -75,
        -22,
        -98,
        -11
      ],
      "24": [
        30,
        -71,
        -106,
        26,
        32,
        -42,
        -50,
        104,
        56,
        64,
        -48,
        -61,
        -4,
        -8,
        -104,
        -12
      ],
      "25": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "26": [
        7,
        56,
        70,
        0,
        36,
        -58,
        -42,
        33,
        -104,
        34,
        -127,
        67,
        -99,
        96,
        -105,
        -12
      ]
    }
  }
}
```

Another example where we brute-force "true" search without a retrieval step using nearestNeighbor or other filters.

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
print(json.dumps(response.hits[0], indent=2))
```

from vespa.io import VespaQueryResponse import json response: VespaQueryResponse = app.query( yql="select * from doc where true", ranking="default", body={ "presentation.format.tensors": "short-value", "input.query(qt)": float_query_token_vectors(query_vectors), }, ) assert response.is_successful() print(json.dumps(response.hits[0], indent=2))

In \[ \]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()
