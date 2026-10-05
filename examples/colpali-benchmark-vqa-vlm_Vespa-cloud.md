# ColPali Ranking Experiments on DocVQA[¶](#colpali-ranking-experiments-on-docvqa)

This notebook demonstrates how to reproduce the ColPali results on [DocVQA](https://huggingface.co/datasets/vidore/docvqa_test_subsampled) with Vespa. The dataset consists of PDF documents with questions and answers.

We demonstrate how we can binarize the patch embeddings and replace the float MaxSim scoring with a `hamming` based MaxSim without much loss in ranking accuracy but with a significant speedup (close to 4x) and reducing the memory (and storage) requirements by 32x.

In this notebook, we represent one PDF page as one vespa document. See other notebooks for more information about using ColPali with Vespa:

- [Scaling ColPALI (VLM) Retrieval](https://vespa-engine.github.io/pyvespa/examples/simplified-retrieval-with-colpali-vlm_Vespa-cloud.ipynb)
- [Vespa 🤝 ColPali: Efficient Document Retrieval with Vision Language Models](https://vespa-engine.github.io/pyvespa/examples/colpali-document-retrieval-vision-language-models-cloud.ipynb)

Install dependencies:

In \[ \]:

Copied!

```
!pip3 install transformers==4.51.3 accelerate pyvespa vespacli requests numpy scipy ir_measures pillow datasets
```

!pip3 install transformers==4.51.3 accelerate pyvespa vespacli requests numpy scipy ir_measures pillow datasets

In \[ \]:

Copied!

```
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor
```

import torch from torch.utils.data import DataLoader from tqdm import tqdm from PIL import Image from transformers import ColPaliForRetrieval, ColPaliProcessor

### Load the model[¶](#load-the-model)

Load the model, also choose the correct device and model weights.

Choose the right device to run the model on.

In \[ \]:

Copied!

```
# Load model (bfloat16 support is limited; fallback to float32 if needed)
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"  # For Apple Silicon devices
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
```

# Load model (bfloat16 support is limited; fallback to float32 if needed)

device = "cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available(): device = "mps" # For Apple Silicon devices dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

Load the base model and the adapter.

In \[ \]:

Copied!

```
model_name = "vidore/colpali-v1.2-hf"
model = ColPaliForRetrieval.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map=device,  # "cpu", "cuda", or "mps" for Apple Silicon
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)
```

model_name = "vidore/colpali-v1.2-hf" model = ColPaliForRetrieval.from_pretrained( model_name, torch_dtype=dtype, device_map=device, # "cpu", "cuda", or "mps" for Apple Silicon ).eval() processor = ColPaliProcessor.from_pretrained(model_name)

### The ViDoRe Benchmark[¶](#the-vidore-benchmark)

We load the DocVQA test set, a subset of the ViDoRe dataset It has 500 pages and a question per page. The task is retrieve the page across the 500 indexed pages.

In \[5\]:

Copied!

```
from datasets import load_dataset

ds = load_dataset("vidore/docvqa_test_subsampled", split="test")
```

from datasets import load_dataset ds = load_dataset("vidore/docvqa_test_subsampled", split="test")

Now we use the ColPali model to generate embeddings for the images in the dataset. We use a dataloader to process each image and store the embeddings in a list.

Batch size 4 requires a GPU with 16GB of memory and fits into a T4 GPU. If you have a smaller GPU, you can reduce the batch size to 2.

In \[ \]:

Copied!

```
dataloader = DataLoader(
    ds["image"],
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: processor(images=x, return_tensors="pt"),
)
embeddings = []
for batch_doc in tqdm(dataloader):
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
        embeddings_doc = model(**batch_doc).embeddings
        embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
```

dataloader = DataLoader( ds["image"], batch_size=4, shuffle=False, collate_fn=lambda x: processor(images=x, return_tensors="pt"), ) embeddings = [] for batch_doc in tqdm(dataloader): with torch.no_grad(): batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()} embeddings_doc = model(\*\*batch_doc).embeddings embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

```
100%|██████████| 125/125 [29:29<00:00, 14.16s/it]
```

Generate embeddings for the queries in the dataset.

In \[ \]:

Copied!

```
dummy_image = Image.new("RGB", (448, 448), (255, 255, 255))
dataloader = DataLoader(
    ds["query"],
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: processor(text=x, return_tensors="pt"),
)
query_embeddings = []
for batch_query in tqdm(dataloader):
    with torch.no_grad():
        batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query).embeddings
        query_embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))
```

dummy_image = Image.new("RGB", (448, 448), (255, 255, 255)) dataloader = DataLoader( ds["query"], batch_size=1, shuffle=False, collate_fn=lambda x: processor(text=x, return_tensors="pt"), ) query_embeddings = [] for batch_query in tqdm(dataloader): with torch.no_grad(): batch_query = {k: v.to(model.device) for k, v in batch_query.items()} embeddings_query = model(\*\*batch_query).embeddings query_embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))

```
100%|██████████| 500/500 [01:45<00:00,  4.72it/s]
```

Now we have all the embeddings. We'll define two helper functions to perform binarization (BQ) and also packing float values to shorter hex representation in JSON. Both save bandwidth and improve feed performance.

In \[8\]:

Copied!

```
import struct
import numpy as np


def binarize_tensor(tensor: torch.Tensor) -> str:
    """
    Binarize a floating-point 1-d tensor by thresholding at zero
    and packing the bits into bytes. Returns the hex str representation of the bytes.
    """
    if not tensor.is_floating_point():
        raise ValueError("Input tensor must be of floating-point type.")
    return (
        np.packbits(np.where(tensor > 0, 1, 0), axis=0).astype(np.int8).tobytes().hex()
    )
```

import struct import numpy as np def binarize_tensor(tensor: torch.Tensor) -> str: """ Binarize a floating-point 1-d tensor by thresholding at zero and packing the bits into bytes. Returns the hex str representation of the bytes. """ if not tensor.is_floating_point(): raise ValueError("Input tensor must be of floating-point type.") return ( np.packbits(np.where(tensor > 0, 1, 0), axis=0).astype(np.int8).tobytes().hex() )

In \[9\]:

Copied!

```
def tensor_to_hex_bfloat16(tensor: torch.Tensor) -> str:
    if not tensor.is_floating_point():
        raise ValueError("Input tensor must be of float32 type.")

    def float_to_bfloat16_hex(f: float) -> str:
        packed_float = struct.pack("=f", f)
        bfloat16_bits = struct.unpack("=H", packed_float[2:])[0]
        return format(bfloat16_bits, "04X")

    hex_list = [float_to_bfloat16_hex(float(val)) for val in tensor.flatten()]
    return "".join(hex_list)
```

def tensor_to_hex_bfloat16(tensor: torch.Tensor) -> str: if not tensor.is_floating_point(): raise ValueError("Input tensor must be of float32 type.") def float_to_bfloat16_hex(f: float) -> str: packed_float = struct.pack("=f", f) bfloat16_bits = struct.unpack("=H", packed_float[2:])[0] return format(bfloat16_bits, "04X") hex_list = [float_to_bfloat16_hex(float(val)) for val in tensor.flatten()] return "".join(hex_list)

### Patch Vector pooling[¶](#patch-vector-pooling)

This reduces the number of patch embeddings by a factor of 3, meaning that we go from 1030 patch vectors to 343 patch vectors. This reduces both the memory and the number of dotproducts we need to calculate. This function is not in use in this notebook, but it is included for reference.

In \[ \]:

Copied!

```
from scipy.cluster.hierarchy import fcluster, linkage
from typing import Dict, List


def pool_embeddings(embeddings: torch.Tensor, pool_factor=3) -> torch.Tensor:
    """
    pool embeddings using hierarchical clustering to reduce the number of patch embeddings.
    Adapted from https://github.com/illuin-tech/vidore-benchmark/blob/e3b4f456d50271c69bce3d2c23131f5245d0c270/src/vidore_benchmark/compression/token_pooling.py#L32
    Inspired by https://www.answer.ai/posts/colbert-pooling.html
    """

    pooled_embeddings = []
    token_length = embeddings.size(0)

    if token_length == 1:
        raise ValueError("The input tensor must have more than one token.")
    embeddings.to(device)

    similarities = torch.mm(embeddings, embeddings.t())
    if similarities.dtype == torch.bfloat16:
        similarities = similarities.to(torch.float16)
    similarities = 1 - similarities.cpu().numpy()

    Z = linkage(similarities, metric="euclidean", method="ward")  # noqa: N806
    max_clusters = max(token_length // pool_factor, 1)
    cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")

    cluster_id_to_indices: Dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for cluster_id in range(1, max_clusters + 1):
            cluster_indices = torch.where(torch.tensor(cluster_labels == cluster_id))[0]
            cluster_id_to_indices[cluster_id] = cluster_indices

            if cluster_indices.numel() > 0:
                pooled_embedding = embeddings[cluster_indices].mean(dim=0)
                pooled_embedding = torch.nn.functional.normalize(
                    pooled_embedding, p=2, dim=-1
                )
                pooled_embeddings.append(pooled_embedding)

        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)

    return pooled_embeddings
```

from scipy.cluster.hierarchy import fcluster, linkage from typing import Dict, List def pool_embeddings(embeddings: torch.Tensor, pool_factor=3) -> torch.Tensor: """ pool embeddings using hierarchical clustering to reduce the number of patch embeddings. Adapted from https://github.com/illuin-tech/vidore-benchmark/blob/e3b4f456d50271c69bce3d2c23131f5245d0c270/src/vidore_benchmark/compression/token_pooling.py#L32 Inspired by https://www.answer.ai/posts/colbert-pooling.html """ pooled_embeddings = [] token_length = embeddings.size(0) if token_length == 1: raise ValueError("The input tensor must have more than one token.") embeddings.to(device) similarities = torch.mm(embeddings, embeddings.t()) if similarities.dtype == torch.bfloat16: similarities = similarities.to(torch.float16) similarities = 1 - similarities.cpu().numpy() Z = linkage(similarities, metric="euclidean", method="ward") # noqa: N806 max_clusters = max(token_length // pool_factor, 1) cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust") cluster_id_to_indices: Dict[int, torch.Tensor] = {} with torch.no_grad(): for cluster_id in range(1, max_clusters + 1): cluster_indices = torch.where(torch.tensor(cluster_labels == cluster_id))[0] cluster_id_to_indices[cluster_id] = cluster_indices if cluster_indices.numel() > 0: pooled_embedding = embeddings[cluster_indices].mean(dim=0) pooled_embedding = torch.nn.functional.normalize( pooled_embedding, p=2, dim=-1 ) pooled_embeddings.append(pooled_embedding) pooled_embeddings = torch.stack(pooled_embeddings, dim=0) return pooled_embeddings

Create the Vespa feed format. We use hex formats for mixed tensors [doc](https://docs.vespa.ai/en/reference/document-json-format.html#tensor).

In \[12\]:

Copied!

```
vespa_docs = []

for row, embedding in zip(ds, embeddings):
    embedding_full = dict()
    embedding_binary = dict()
    # You can experiment with pooling if you want to reduce the number of embeddings
    # pooled_embedding = pool_embeddings(embedding, pool_factor=2) # reduce the number of embeddings by a factor of 2
    for j, emb in enumerate(embedding):
        embedding_full[j] = tensor_to_hex_bfloat16(emb)
        embedding_binary[j] = binarize_tensor(emb)
    vespa_doc = {
        "id": row["docId"],
        "embedding": embedding_full,
        "binary_embedding": embedding_binary,
    }
    vespa_docs.append(vespa_doc)
```

vespa_docs = [] for row, embedding in zip(ds, embeddings): embedding_full = dict() embedding_binary = dict()

# You can experiment with pooling if you want to reduce the number of embeddings

# pooled_embedding = pool_embeddings(embedding, pool_factor=2) # reduce the number of embeddings by a factor of 2

for j, emb in enumerate(embedding): embedding_full[j] = tensor_to_hex_bfloat16(emb) embedding_binary[j] = binarize_tensor(emb) vespa_doc = { "id": row["docId"], "embedding": embedding_full, "binary_embedding": embedding_binary, } vespa_docs.append(vespa_doc)

### Configure Vespa[¶](#configure-vespa)

[PyVespa](https://vespa-engine.github.io/pyvespa/) helps us build the [Vespa application package](https://docs.vespa.ai/en/application-packages.html). A Vespa application package consists of configuration files, schemas, models, and code (plugins).

First, we define a [Vespa schema](https://docs.vespa.ai/en/schemas.html) with the fields we want to store and their type. This is a simple schema, which is all we need to evaluate the effectiveness of the model.

In \[14\]:

Copied!

```
from vespa.package import Schema, Document, Field

colpali_schema = Schema(
    name="pdf_page",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary", "attribute"]),
            Field(
                name="embedding",
                type="tensor<bfloat16>(patch{}, v[128])",
                indexing=["attribute"],
            ),
            Field(
                name="binary_embedding",
                type="tensor<int8>(patch{}, v[16])",
                indexing=["attribute"],
            ),
        ]
    ),
)
```

from vespa.package import Schema, Document, Field colpali_schema = Schema( name="pdf_page", document=Document( fields=\[ Field(name="id", type="string", indexing=["summary", "attribute"]), Field( name="embedding", type="tensor<bfloat16>(patch{}, v[128])", indexing=["attribute"], ), Field( name="binary_embedding", type="tensor<int8>(patch{}, v[16])", indexing=["attribute"], ), \] ), )

In \[15\]:

Copied!

```
from vespa.package import ApplicationPackage

vespa_app_name = "visionragtest"
vespa_application_package = ApplicationPackage(
    name=vespa_app_name, schema=[colpali_schema]
)
```

from vespa.package import ApplicationPackage vespa_app_name = "visionragtest" vespa_application_package = ApplicationPackage( name=vespa_app_name, schema=[colpali_schema] )

Now we define how we want to rank the pages. We have 4 ranking models that we want to evaluate. These are all MaxSim variants but with various precision trade-offs.

1. **float-float** A regular MaxSim implementation that uses the float representation of both query and page embeddings.
1. **float-binary** Use the binarized representation of the page embeddings and where we unpack it into float representation. The query representation is still float.
1. **binary-binary** Use the binarized representation of the doc embeddings and the query embeddings and replaces the dot product with inverted hamming distance.
1. **phased** This uses the binary-binary in a first-phase, and then re-ranks using the float-binary representation. Only top 20 pages are re-ranked (This can be overriden in the query request as well).

In \[17\]:

Copied!

```
from vespa.package import RankProfile, Function, FirstPhaseRanking, SecondPhaseRanking

colpali_profile = RankProfile(
    name="float-float",
    # We define both the float and binary query inputs here; the rest of the profiles inherit these inputs
    inputs=[
        ("query(qtb)", "tensor<int8>(querytoken{}, v[16])"),
        ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
    ],
    functions=[
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * cell_cast(attribute(embedding), float), v
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        )
    ],
    first_phase=FirstPhaseRanking(expression="max_sim"),
)

colpali_binary_profile = RankProfile(
    name="float-binary",
    inherits="float-float",
    functions=[
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(binary_embedding)), v
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        )
    ],
    first_phase=FirstPhaseRanking(expression="max_sim"),
)

colpali_hamming_profile = RankProfile(
    name="binary-binary",
    inherits="float-float",
    functions=[
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        1/(1+ sum(
                            hamming(query(qtb), attribute(binary_embedding)),v
                        )),
                        max, patch
                    ),
                    querytoken
                )
            """,
        )
    ],
    first_phase=FirstPhaseRanking(expression="max_sim"),
)

colpali__phased_hamming_profile = RankProfile(
    name="phased",
    inherits="float-float",
    functions=[
        Function(
            name="max_sim_hamming",
            expression="""
                sum(
                    reduce(
                        1/(1+ sum(
                            hamming(query(qtb), attribute(binary_embedding)),v
                        )),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(binary_embedding)), v
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
    ],
    first_phase=FirstPhaseRanking(expression="max_sim_hamming"),
    second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=20),
)


colpali_schema.add_rank_profile(colpali_profile)
colpali_schema.add_rank_profile(colpali_binary_profile)
colpali_schema.add_rank_profile(colpali_hamming_profile)
colpali_schema.add_rank_profile(colpali__phased_hamming_profile)
```

from vespa.package import RankProfile, Function, FirstPhaseRanking, SecondPhaseRanking colpali_profile = RankProfile( name="float-float",

# We define both the float and binary query inputs here; the rest of the profiles inherit these inputs

inputs=\[ ("query(qtb)", "tensor<int8>(querytoken{}, v[16])"), ("query(qt)", "tensor<float>(querytoken{}, v[128])"), \], functions=[ Function( name="max_sim", expression=""" sum( reduce( sum( query(qt) * cell_cast(attribute(embedding), float), v ), max, patch ), querytoken ) """, ) ], first_phase=FirstPhaseRanking(expression="max_sim"), ) colpali_binary_profile = RankProfile( name="float-binary", inherits="float-float", functions=[ Function( name="max_sim", expression=""" sum( reduce( sum( query(qt) * unpack_bits(attribute(binary_embedding)), v ), max, patch ), querytoken ) """, ) ], first_phase=FirstPhaseRanking(expression="max_sim"), ) colpali_hamming_profile = RankProfile( name="binary-binary", inherits="float-float", functions=[ Function( name="max_sim", expression=""" sum( reduce( 1/(1+ sum( hamming(query(qtb), attribute(binary_embedding)),v )), max, patch ), querytoken ) """, ) ], first_phase=FirstPhaseRanking(expression="max_sim"), ) colpali\_\_phased_hamming_profile = RankProfile( name="phased", inherits="float-float", functions=[ Function( name="max_sim_hamming", expression=""" sum( reduce( 1/(1+ sum( hamming(query(qtb), attribute(binary_embedding)),v )), max, patch ), querytoken ) """, ), Function( name="max_sim", expression=""" sum( reduce( sum( query(qt) * unpack_bits(attribute(binary_embedding)), v ), max, patch ), querytoken ) """, ), ], first_phase=FirstPhaseRanking(expression="max_sim_hamming"), second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=20), ) colpali_schema.add_rank_profile(colpali_profile) colpali_schema.add_rank_profile(colpali_binary_profile) colpali_schema.add_rank_profile(colpali_hamming_profile) colpali_schema.add_rank_profile(colpali\_\_phased_hamming_profile)

### Deploy to Vespa Cloud[¶](#deploy-to-vespa-cloud)

With the configured application, we can deploy it to [Vespa Cloud](https://cloud.vespa.ai/en/).

`PyVespa` supports deploying apps to the [development zone](https://cloud.vespa.ai/en/reference/environments#dev-and-perf).

> Note: Deployments to dev and perf expire after 7 days of inactivity, i.e., 7 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 7 more days.

To deploy the application to Vespa Cloud we need to create a tenant in the Vespa Cloud:

Create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/) (unless you already have one). This step requires a Google or GitHub account, and will start your [free trial](https://cloud.vespa.ai/en/free-trial). Make note of the tenant name, it is used in the next steps.

In \[ \]:

Copied!

```
from vespa.deployment import VespaCloud
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Replace with your tenant name from the Vespa Cloud Console
tenant_name = "vespa-team"

key = os.getenv("VESPA_TEAM_API_KEY", None)
if key is not None:
    key = key.replace(r"\n", "\n")  # To parse key correctly

vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=vespa_app_name,
    key_content=key,  # Key is only used for CI/CD testing of this notebook. Can be removed if logging in interactively
    application_package=vespa_application_package,
)
```

from vespa.deployment import VespaCloud import os os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Replace with your tenant name from the Vespa Cloud Console

tenant_name = "vespa-team" key = os.getenv("VESPA_TEAM_API_KEY", None) if key is not None: key = key.replace(r"\\n", "\\n") # To parse key correctly vespa_cloud = VespaCloud( tenant=tenant_name, application=vespa_app_name, key_content=key, # Key is only used for CI/CD testing of this notebook. Can be removed if logging in interactively application_package=vespa_application_package, )

Now deploy the app to Vespa Cloud dev zone.

The first deployment typically takes 2 minutes until the endpoint is up.

In \[ \]:

Copied!

```
from vespa.application import Vespa

app: Vespa = vespa_cloud.deploy()
```

from vespa.application import Vespa app: Vespa = vespa_cloud.deploy()

This example uses the asynchronous feed method and feeds one document at a time.

In \[23\]:

Copied!

```
from vespa.io import VespaResponse

async with app.asyncio(connections=1, timeout=180) as session:
    for doc in tqdm(vespa_docs):
        response: VespaResponse = await session.feed_data_point(
            data_id=doc["id"], fields=doc, schema="pdf_page"
        )
        if not response.is_successful():
            print(response.json())
```

from vespa.io import VespaResponse async with app.asyncio(connections=1, timeout=180) as session: for doc in tqdm(vespa_docs): response: VespaResponse = await session.feed_data_point( data_id=doc["id"], fields=doc, schema="pdf_page" ) if not response.is_successful(): print(response.json())

```
100%|██████████| 500/500 [01:13<00:00,  6.77it/s]
```

### Run queries and evaluate effectiveness[¶](#run-queries-and-evaluate-effectiveness)

We use ir_measures to evaluate the effectiveness of the retrieval model.

In \[24\]:

Copied!

```
from ir_measures import calc_aggregate, nDCG, ScoredDoc, Qrel
```

from ir_measures import calc_aggregate, nDCG, ScoredDoc, Qrel

A simple routine for querying Vespa. Note that we send both vector representations in the query independently of the ranking method used, this for simplicity. Not all the ranking models we evaluate need both representations.

In \[32\]:

Copied!

```
from vespa.io import VespaQueryResponse
from vespa.application import VespaAsync


async def get_vespa_response(
    embedding: torch.Tensor,
    qid: str,
    session: VespaAsync,
    depth=20,
    profile="float-float",
) -> List[ScoredDoc]:
    # The query tensor api does not support hex formats yet
    float_embedding = {index: vector.tolist() for index, vector in enumerate(embedding)}
    binary_embedding = {
        index: np.packbits(np.where(vector > 0, 1, 0), axis=0).astype(np.int8).tolist()
        for index, vector in enumerate(embedding)
    }
    response: VespaQueryResponse = await session.query(
        yql="select id from pdf_page where true",  # brute force search, rank all pages
        ranking=profile,
        hits=5,
        timeout=10,
        body={
            "input.query(qt)": float_embedding,
            "input.query(qtb)": binary_embedding,
            "ranking.rerankCount": depth,
        },
    )
    assert response.is_successful()
    scored_docs = []
    for hit in response.hits:
        doc_id = hit["fields"]["id"]
        score = hit["relevance"]
        scored_docs.append(ScoredDoc(qid, doc_id, score))
    return scored_docs
```

from vespa.io import VespaQueryResponse from vespa.application import VespaAsync async def get_vespa_response( embedding: torch.Tensor, qid: str, session: VespaAsync, depth=20, profile="float-float", ) -> List\[ScoredDoc\]:

# The query tensor api does not support hex formats yet

float_embedding = {index: vector.tolist() for index, vector in enumerate(embedding)} binary_embedding = { index: np.packbits(np.where(vector > 0, 1, 0), axis=0).astype(np.int8).tolist() for index, vector in enumerate(embedding) } response: VespaQueryResponse = await session.query( yql="select id from pdf_page where true", # brute force search, rank all pages ranking=profile, hits=5, timeout=10, body={ "input.query(qt)": float_embedding, "input.query(qtb)": binary_embedding, "ranking.rerankCount": depth, }, ) assert response.is_successful() scored_docs = [] for hit in response.hits: doc_id = hit["fields"]["id"] score = hit["relevance"] scored_docs.append(ScoredDoc(qid, doc_id, score)) return scored_docs

Run a test query first..

In \[28\]:

Copied!

```
async with app.asyncio() as session:
    for profile in ["float-float", "float-binary", "binary-binary", "phased"]:
        print(
            await get_vespa_response(
                query_embeddings[0], profile, session, profile=profile
            )
        )
```

async with app.asyncio() as session: for profile in \["float-float", "float-binary", "binary-binary", "phased"\]: print( await get_vespa_response( query_embeddings[0], profile, session, profile=profile ) )

```
[ScoredDoc(query_id='float-float', doc_id='4720', score=16.292504370212555), ScoredDoc(query_id='float-float', doc_id='4858', score=13.315170526504517), ScoredDoc(query_id='float-float', doc_id='14686', score=12.212152108550072), ScoredDoc(query_id='float-float', doc_id='4846', score=12.002869427204132), ScoredDoc(query_id='float-float', doc_id='864', score=11.308563649654388)]
[ScoredDoc(query_id='float-binary', doc_id='4720', score=82.99432492256165), ScoredDoc(query_id='float-binary', doc_id='4858', score=71.45464742183685), ScoredDoc(query_id='float-binary', doc_id='14686', score=68.46699643135071), ScoredDoc(query_id='float-binary', doc_id='4846', score=64.85357594490051), ScoredDoc(query_id='float-binary', doc_id='2161', score=63.85516130924225)]
[ScoredDoc(query_id='binary-binary', doc_id='4720', score=0.771387243643403), ScoredDoc(query_id='binary-binary', doc_id='4858', score=0.7132036704570055), ScoredDoc(query_id='binary-binary', doc_id='14686', score=0.6979007869958878), ScoredDoc(query_id='binary-binary', doc_id='6087', score=0.6534321829676628), ScoredDoc(query_id='binary-binary', doc_id='2161', score=0.6525899451225996)]
[ScoredDoc(query_id='phased', doc_id='4720', score=82.99432492256165), ScoredDoc(query_id='phased', doc_id='4858', score=71.45464742183685), ScoredDoc(query_id='phased', doc_id='14686', score=68.46699643135071), ScoredDoc(query_id='phased', doc_id='4846', score=64.85357594490051), ScoredDoc(query_id='phased', doc_id='2161', score=63.85516130924225)]
```

Now, run through all of the test queries for each of the ranking models.

In \[29\]:

Copied!

```
qrels = []
profiles = ["float-float", "float-binary", "binary-binary", "phased"]
results = {profile: [] for profile in profiles}
async with app.asyncio(connections=3) as session:
    for row, embedding in zip(tqdm(ds), query_embeddings):
        qrels.append(Qrel(row["questionId"], str(row["docId"]), 1))
        for profile in profiles:
            scored_docs = await get_vespa_response(
                embedding, row["questionId"], session, profile=profile
            )
            results[profile].extend(scored_docs)
```

qrels = [] profiles = ["float-float", "float-binary", "binary-binary", "phased"] results = {profile: [] for profile in profiles} async with app.asyncio(connections=3) as session: for row, embedding in zip(tqdm(ds), query_embeddings): qrels.append(Qrel(row["questionId"], str(row["docId"]), 1)) for profile in profiles: scored_docs = await get_vespa_response( embedding, row["questionId"], session, profile=profile ) results[profile].extend(scored_docs)

```
500it [11:32,  1.39s/it]
```

Calculate the effectiveness of the 4 different models

In \[30\]:

Copied!

```
for profile in profiles:
    score = calc_aggregate([nDCG @ 5], qrels, results[profile])[nDCG @ 5]
    print(f"nDCG@5 for {profile}: {100*score:.2f}")
```

for profile in profiles: score = calc_aggregate([nDCG @ 5], qrels, results[profile])[nDCG @ 5] print(f"nDCG@5 for {profile}: {100\*score:.2f}")

```
nDCG@5 for float-float: 52.37
nDCG@5 for float-binary: 51.64
nDCG@5 for binary-binary: 49.48
nDCG@5 for phased: 51.70
```

This is encouraging as the binary-binary representation is 4x faster than the float-float representation and saves 32x space. We can also largely retain the effectiveness of the float-binary representation by using the phased approach, where we re-rank the top 20 pages from the hamming (binary-binary) version using the float-binary representation. Now we can explore the ranking depth and see how the phased approach performs with different ranking depths.

In \[35\]:

Copied!

```
results = {
    profile: []
    for profile in [
        "phased-rerank-count=5",
        "phased-rerank-count=10",
        "phased-rerank-count=20",
        "phased-rerank-count=40",
    ]
}
async with app.asyncio(connections=3) as session:
    for row, embedding in zip(tqdm(ds), query_embeddings):
        qrels.append(Qrel(row["questionId"], str(row["docId"]), 1))
        for count in [5, 10, 20, 40]:
            scored_docs = await get_vespa_response(
                embedding, row["questionId"], session, profile="phased", depth=count
            )
            results["phased-rerank-count=" + str(count)].extend(scored_docs)
```

results = { profile: [] for profile in [ "phased-rerank-count=5", "phased-rerank-count=10", "phased-rerank-count=20", "phased-rerank-count=40", ] } async with app.asyncio(connections=3) as session: for row, embedding in zip(tqdm(ds), query_embeddings): qrels.append(Qrel(row["questionId"], str(row["docId"]), 1)) for count in \[5, 10, 20, 40\]: scored_docs = await get_vespa_response( embedding, row["questionId"], session, profile="phased", depth=count ) results["phased-rerank-count=" + str(count)].extend(scored_docs)

```
500it [08:18,  1.00it/s]
```

In \[36\]:

Copied!

```
for profile in results.keys():
    score = calc_aggregate([nDCG @ 5], qrels, results[profile])[nDCG @ 5]
    print(f"nDCG@5 for {profile}: {100*score:.2f}")
```

for profile in results.keys(): score = calc_aggregate([nDCG @ 5], qrels, results[profile])[nDCG @ 5] print(f"nDCG@5 for {profile}: {100\*score:.2f}")

```
nDCG@5 for phased-rerank-count=5: 50.77
nDCG@5 for phased-rerank-count=10: 51.58
nDCG@5 for phased-rerank-count=20: 51.70
nDCG@5 for phased-rerank-count=40: 51.64
```

### Conclusion[¶](#conclusion)

The binary representation of the patch embeddings reduces the storage by 32x, and using hamming distance instead of dotproduct saves us about 4x in computation compared to the float-float model or the float-binary model (which only saves storage). Using a re-ranking step with only depth 10, we can improve the effectiveness of the binary-binary model to almost match the float-float MaxSim model. The additional re-ranking step only requires that we pass also the float query embedding version without any additional storage overhead.
