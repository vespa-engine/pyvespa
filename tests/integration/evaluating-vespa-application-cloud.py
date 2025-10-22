# %%
import os
import pandas as pd
import vespa.querybuilder as qb
from datasets import load_dataset
from vespa.application import Vespa
from vespa.deployment import VespaCloud
from vespa.evaluation import VespaMatchEvaluator, VespaEvaluator
from vespa.io import VespaResponse
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

# %%
tenant_name = "vespa-team"
application = "modernbert"
schema_name = "doc"

# %%
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
                        type="tensor<float>(x[768])",
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
                    inputs=[("query(q)", "tensor<float>(x[768])")],
                    first_phase="random",  # TODO: Remove when pyvespa supports empty first_phase
                ),
                RankProfile(
                    name="bm25",
                    inputs=[("query(q)", "tensor<float>(x[768])")],
                    functions=[Function(name="bm25text", expression="bm25(text)")],
                    first_phase="bm25text",
                    match_features=["bm25text"],
                ),
                RankProfile(
                    name="semantic",
                    inputs=[("query(q)", "tensor<float>(x[768])")],
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
                    inputs=[("query(q)", "tensor<float>(x[768])")],
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
                    inputs=[("query(q)", "tensor<float>(x[768])")],
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
            id="modernbert",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    name="transformer-model",
                    args={
                        "url": "https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/onnx/model_quantized.onnx"
                    },
                ),
                Parameter(
                    name="tokenizer-model",
                    args={
                        "url": "https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/tokenizer.json"
                    },
                ),
                Parameter(
                    name="transformer-output",
                    args={},
                    children="token_embeddings",
                ),
                Parameter(
                    name="max-tokens",
                    args={},
                    children="8192",
                ),
                Parameter(
                    name="prepend",
                    args={},
                    children=[
                        Parameter(
                            name="query",
                            args={},
                            children="Represent this sentence for searching relevant passages: ",
                        ),
                        # Parameter(name="document", args={}, children="passage: "),
                    ],
                ),
            ],
        )
    ],
)

# %%
package.to_files("modernbert")

# %%
vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=application,
    key_content=os.getenv("VESPA_TEAM_API_KEY", None),
    application_package=package,
)

# %%
app: Vespa = vespa_cloud.deploy()

# %%

dataset_id = "zeta-alpha-ai/NanoMSMARCO"

dataset = load_dataset(dataset_id, "corpus", split="train", streaming=True)
vespa_feed = dataset.map(
    lambda x: {
        "id": x["_id"],
        "fields": {"text": x["text"], "id": x["_id"]},
    }
)

# %%
query_ds = load_dataset(dataset_id, "queries", split="train")
qrels = load_dataset(dataset_id, "qrels", split="train")

# %%
ids_to_query = dict(zip(query_ds["_id"], query_ds["text"]))

# %%
for idx, (qid, q) in enumerate(ids_to_query.items()):
    print(f"qid: {qid}, query: {q}")
    if idx == 5:
        break

# %%
relevant_docs = dict(zip(qrels["query-id"], qrels["corpus-id"]))

# %%
for idx, (qid, doc_id) in enumerate(relevant_docs.items()):
    print(f"qid: {qid}, doc_id: {doc_id}")
    if idx == 5:
        break


# %%
def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Error when feeding document {id}: {response.get_json()}")


app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)


# %%
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


# %%
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

# %%
results = pd.DataFrame(match_results)
results


# %%
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


# %%
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

# %%
results = pd.DataFrame(all_results)

# %%
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

# %%
results.plot(kind="bar", figsize=(12, 6))

# %%
searchtime = searchtime * 1000
searchtime.plot(kind="bar", figsize=(12, 6)).set(ylabel="time (ms)")

# %%
vespa_cloud.delete()
