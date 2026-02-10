# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""
Voyage AI Vespa Cloud Deployment and Evaluation Script.

This creates an application package with voyage-ai-embedder components,
deploys it to Vespa Cloud, feeds the NanoMSMarco dataset, and evaluates
the float_angular rank profile using VespaEvaluator.
"""

from vespa.package import (
    ApplicationPackage,
    Schema,
    Document,
    Field,
    RankProfile,
    ServicesConfiguration,
)
from vespa.configuration.services import (
    services,
    container,
    content,
    search,
    document_api,
    document_processing,
    component,
    components,
    model,
    api_key_secret_ref,
    dimensions,
    documents,
    document,
    nodes,
    node,
    secrets,
    threadpool,
    threads,
    redundancy,
    transformer_model,
    tokenizer_model,
    pooling_strategy,
    normalize,
    prepend,
    max_tokens,
    query,
)
from vespa.configuration.vt import vt
from vespa.deployment import VespaCloud
from vespa.evaluation import VespaEvaluator
from vespa.io import VespaResponse
import vespa.querybuilder as qb

# Configuration
TENANT_NAME = "thttest04"
APPLICATION_NAME = "voyagetest2"
SCHEMA_NAME = "doc"
SECRET_STORE_VAULT_NAME = "thtvault"
VOYAGE_SECRET_NAME = "voyage_api_key"

FEED_MODEL_ID = "voyage-4-large"
QUERY_MODEL_ID = "voyage-4-nano-int8"

# Define the schema with document fields
schema = Schema(
    name=SCHEMA_NAME,
    document=Document(
        fields=[
            # id field is required for evaluation to match documents with qrels
            Field(name="id", type="string", indexing=["summary", "attribute"]),
            Field(name="text", type="string", indexing=["index", "summary"]),
        ]
    ),
)

# Add synthetic embedding fields (not part of document, computed at indexing time)
# These use the voyage-large embedder to generate embeddings from the text field

# Float embedding field with prenormalized-angular distance metric
schema.add_fields(
    Field(
        name="embedding_float",
        type="tensor<float>(x[2048])",
        indexing=["input text", f"embed {FEED_MODEL_ID}", "attribute"],
        attribute=["distance-metric: prenormalized-angular"],
        is_document_field=False,
    )
)

# Binary int8 embedding field with hamming distance metric
schema.add_fields(
    Field(
        name="embedding_binary_int8",
        type="tensor<int8>(x[256])",  # 2048 // 8 = 256
        indexing=["input text", f"embed {FEED_MODEL_ID}", "attribute"],
        attribute=["distance-metric: hamming"],
        is_document_field=False,
    )
)

# Int8 embedding field with prenormalized-angular distance metric
schema.add_fields(
    Field(
        name="embedding_int8",
        type="tensor<int8>(x[2048])",
        indexing=["input text", f"embed {FEED_MODEL_ID}", "attribute"],
        attribute=["distance-metric: prenormalized-angular"],
        is_document_field=False,
    )
)

# Add rank profiles for each embedding type
schema.add_rank_profile(
    RankProfile(
        name="float_angular",
        inputs=[("query(embedding_float)", "tensor<float>(x[2048])")],
        first_phase="closeness(embedding_float)",
        summary_features=[
            "query(embedding_float)",
            "attribute(embedding_float)",
        ],
    )
)

schema.add_rank_profile(
    RankProfile(
        name="binary_int8",
        inputs=[("query(embedding_binary_int8)", "tensor<int8>(x[256])")],
        first_phase="closeness(embedding_binary_int8)",
        summary_features=[
            "query(embedding_binary_int8)",
            "attribute(embedding_binary_int8)",
        ],
    )
)

schema.add_rank_profile(
    RankProfile(
        name="int8_angular",
        inputs=[("query(embedding_int8)", "tensor<int8>(x[1024])")],
        first_phase="closeness(embedding_int8)",
        summary_features=[
            "query(embedding_int8)",
            "attribute(embedding_int8)",
        ],
    )
)
# <document-processing>
#     <!-- Docproc worker thread pool -->
#     <threadpool>
#         <!-- 2 threads per vcpu => 16 threads on 8 vcpu -->
#         <threads>2</threads>
#     </threadpool>
# </document-processing>

# Define services configuration with Voyage AI embedders
services_config = ServicesConfiguration(
    application_name=APPLICATION_NAME,
    services_config=services(
        container(id=f"{APPLICATION_NAME}_container", version="1.0")(
            secrets(
                vt(tag="apiKey", vault=SECRET_STORE_VAULT_NAME, name=VOYAGE_SECRET_NAME)
            ),
            search(),
            document_api(),
            document_processing(threadpool(threads("1000"))),
            # Voyage AI nano embedder (used for query embedding)
            # <component id="my-embedder-id" type="hugging-face-embedder">
            #     <transformer-model model-id="voyage-4-nano-int8"/>
            #     <max-tokens>32768</max-tokens>
            #     <pooling-strategy>mean</pooling-strategy>
            #     <normalize>true</normalize>
            #     <prepend>
            #         <query>Represent the query for retrieving supporting documents: </query>
            #     </prepend>
            # </component>
            components(
                component(id="voyage-4-nano-int8", type_="hugging-face-embedder")(
                    transformer_model(model_id="voyage-4-nano-int8"),
                    tokenizer_model(model_id="voyage-4-nano-vocab"),
                    max_tokens("32768"),
                    pooling_strategy("mean"),
                    normalize("true"),
                    prepend(
                        query(
                            "Represent the query for retrieving supporting documents: "
                        )
                    ),
                ),
                # Voyage AI lite embedder
                component(id="voyage-4-lite", type_="voyage-ai-embedder")(
                    model("voyage-4-lite"),
                    api_key_secret_ref("apiKey"),
                    dimensions("2048"),
                ),
                # Voyage AI large embedder (used by the embedding fields)
                component(id="voyage-4-large", type_="voyage-ai-embedder")(
                    model("voyage-4-large"),
                    api_key_secret_ref("apiKey"),
                    dimensions("2048"),
                ),
            ),
        ),
        content(id=f"{APPLICATION_NAME}_content", version="1.0")(
            redundancy("1"),
            documents(document(type_="doc", mode="index")),
            nodes(node(distribution_key="0", hostalias="node1")),
        ),
    ),
)

# Create the application package
app_package = ApplicationPackage(
    name=APPLICATION_NAME,
    schema=[schema],
    services_config=services_config,
)


def callback(response: VespaResponse, id: str):
    """Callback function for feed operations."""
    if not response.is_successful():
        print(f"Error when feeding document {id}: {response.get_json()}")


def semantic_query_fn(query_text: str, top_k: int) -> dict:
    """Query function for semantic search using Voyage AI embeddings."""
    return {
        "yql": str(
            qb.select("*")
            .from_(SCHEMA_NAME)
            .where(
                qb.nearestNeighbor(
                    field="embedding_float",
                    query_vector="embedding_float",
                    annotations={"targetHits": 1000},
                )
            )
        ),
        "ranking": "float_angular",
        "input.query(embedding_float)": f'embed({QUERY_MODEL_ID}, "{query_text}")',
        "hits": top_k,
    }


if __name__ == "__main__":
    from datasets import load_dataset

    # Dump application package for inspection
    app_package.to_files("output/")

    # 1. Deploy to Vespa Cloud
    print("Deploying to Vespa Cloud...")
    vespa_cloud = VespaCloud(
        tenant=TENANT_NAME,
        application=APPLICATION_NAME,
        key_location=f"/Users/thomas/.vespa/{TENANT_NAME}.api-key.pem",
        application_package=app_package,
    )
    app = vespa_cloud.deploy()

    # 2. Load and feed NanoMSMarco dataset
    print("Loading and feeding NanoMSMarco dataset...")
    dataset_id = "zeta-alpha-ai/NanoMSMARCO"
    dataset = load_dataset(dataset_id, "corpus", split="train", streaming=True)
    vespa_feed = dataset.map(
        lambda x: {
            "id": x["_id"],
            "fields": {"text": x["text"], "id": x["_id"]},
        }
    )
    app.feed_iterable(
        vespa_feed, schema=SCHEMA_NAME, namespace="tutorial", callback=callback
    )

    # 3. Load evaluation data
    print("Loading evaluation data...")
    query_ds = load_dataset(dataset_id, "queries", split="train")
    qrels = load_dataset(dataset_id, "qrels", split="train")
    ids_to_query = dict(zip(query_ds["_id"], query_ds["text"]))
    relevant_docs = dict(zip(qrels["query-id"], qrels["corpus-id"]))

    # 4. Run evaluation
    print("Running evaluation...")
    evaluator = VespaEvaluator(
        queries=ids_to_query,
        relevant_docs=relevant_docs,
        vespa_query_fn=semantic_query_fn,
        app=app,
        name="voyage-float-angular",
        id_field="id",
        write_csv=True,
    )
    results = evaluator.run()

    # 5. Print results
    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # 6. Optional cleanup (uncomment to delete the application)
    # print("Cleaning up...")
    # vespa_cloud.delete()
