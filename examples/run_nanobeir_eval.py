#!/usr/bin/env python3
"""
NanoBEIR Evaluation Runner

This script demonstrates how to run a complete evaluation workflow using the
NanoBEIR dataset with multiple embedding models. It creates a Vespa application,
feeds documents, runs evaluation queries, and saves results to CSV files.
"""

import os
import pandas as pd
import vespa.querybuilder as qb
from datasets import load_dataset
from vespa.application import Vespa
from vespa.deployment import VespaCloud, VespaDocker
from vespa.evaluation import VespaMatchEvaluator, VespaEvaluator
from vespa.io import VespaResponse
from vespa.evaluation import create_evaluation_package, get_model_config, ModelConfig
from enum import Enum

# Configuration
TENANT_NAME = os.getenv("VESPA_TENANT_NAME", "vespa-team")
APPLICATION = "nanobeireval"
SCHEMA_NAME = "doc"
DATASET_ID = "zeta-alpha-ai/NanoMSMARCO"


class DeployTarget(Enum):
    VESPA_CLOUD = "vespa_cloud"
    LOCAL = "local"


TARGET = DeployTarget.LOCAL

# Models to evaluate - you can modify this list
# Can be:
#  - Predefined model names (strings): "e5-small-v2", "nomic-ai-modernbert", etc.
#  - Custom ModelConfig objects for models not in the predefined list
# Example with custom config:
# MODELS = [
#     "e5-small-v2",
#     ModelConfig(
#         model_id="custom-model",
#         embedding_dim=384,
#         binarized=False,
#         query_prepend="query: ",
#         document_prepend="document: ",
#     )
# ]
kalm_model = ModelConfig(
    model_id="kalm",
    model_url="https://huggingface.co/thomasht86/KaLM-embedding-multilingual-mini-instruct-v2.5-ONNX/resolve/main/onnx/model_int8.onnx",
    tokenizer_url="https://huggingface.co/thomasht86/KaLM-embedding-multilingual-mini-instruct-v2.5-ONNX/resolve/main/tokenizer.json",
    transformer_output="token_embeddings",
    embedding_dim=896,
    binarized=False,
    query_prepend="Instruct: Given a query, retrieve documents that answer the query \n Query: ",
)
# 'https://data.vespa-cloud.com/onnx_models/e5-small-v2/model.onnx'
# 'https://data.vespa-cloud.com/onnx_models/e5-small-v2/tokenizer.json'
e5_small_v2 = ModelConfig(
    model_id="e5_small_v2",
    model_url="https://data.vespa-cloud.com/onnx_models/e5-small-v2/model.onnx",
    tokenizer_url="https://data.vespa-cloud.com/onnx_models/e5-small-v2/tokenizer.json",
    embedding_dim=384,
    binarized=False,
    max_tokens=512,
    query_prepend="query: ",
    document_prepend="passage: ",
)

MODELS = [e5_small_v2]


def feed_data(app: Vespa, dataset_id: str, schema_name: str):
    """
    Load and feed the NanoBEIR dataset to Vespa.

    Args:
        app: Vespa application instance
        dataset_id: HuggingFace dataset identifier
        schema_name: Name of the Vespa schema
    """
    print(f"\nLoading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id, "corpus", split="train", streaming=True)

    vespa_feed = dataset.map(
        lambda x: {
            "id": x["_id"],
            "fields": {"text": x["text"], "id": x["_id"]},
        }
    )

    def callback(response: VespaResponse, id: str):
        if not response.is_successful():
            print(f"Error when feeding document {id}: {response.get_json()}")

    print("Feeding documents to Vespa...")
    app.feed_iterable(
        vespa_feed,
        schema=schema_name,
        namespace="nanobeir",
        callback=callback,
    )
    print("Feeding complete!")


def load_queries_and_qrels(dataset_id: str):
    """
    Load queries and relevance judgments from the dataset.

    Args:
        dataset_id: HuggingFace dataset identifier

    Returns:
        Tuple of (queries dict, relevant_docs dict)
    """
    print("\nLoading queries and relevance judgments...")
    query_ds = load_dataset(dataset_id, "queries", split="train")
    qrels = load_dataset(dataset_id, "qrels", split="train")

    queries = dict(zip(query_ds["_id"], query_ds["text"]))
    relevant_docs = dict(zip(qrels["query-id"], qrels["corpus-id"]))

    print(f"Loaded {len(queries)} queries and {len(relevant_docs)} relevance judgments")
    return queries, relevant_docs


def create_query_functions(model_configs, schema_name: str):
    """
    Create query functions for different retrieval strategies.

    Args:
        model_configs: List of ModelConfig objects
        schema_name: Name of the Vespa schema

    Returns:
        Dictionary mapping strategy names to query functions
    """
    is_multi_model = len(model_configs) > 1
    query_functions = {}

    for config in model_configs:
        # Determine naming based on single vs multi-model setup
        if is_multi_model:
            embedding_field = f"embedding_{config.component_id}"
            query_tensor = f"q_{config.component_id}"
            profile_suffix = f"_{config.component_id}"
            model_label = f"_{config.component_id}"
        else:
            embedding_field = "embedding"
            query_tensor = "q"
            profile_suffix = ""
            model_label = ""

        # Match strategies (for VespaMatchEvaluator)
        def make_semantic_match_fn(embedding_field, query_tensor, embedder_id):
            def semantic_match_query_fn(query_text: str, top_k: int) -> dict:
                return {
                    "yql": str(
                        qb.select("*")
                        .from_(schema_name)
                        .where(
                            qb.nearestNeighbor(
                                field=embedding_field,
                                query_vector=query_tensor,
                                annotations={"targetHits": 100},
                            )
                        )
                    ),
                    "query": query_text,
                    "ranking": "match-only",
                    f"input.query({query_tensor})": f"embed({embedder_id}, '{query_text}')",
                }

            return semantic_match_query_fn

        def make_weakand_match_fn(embedder_id):
            def weakand_match_query_fn(query_text: str, top_k: int) -> dict:
                return {
                    "yql": str(
                        qb.select("*")
                        .from_(schema_name)
                        .where(qb.userQuery(query_text))
                    ),
                    "query": query_text,
                    "ranking": "match-only",
                    "input.query(q)": f"embed({embedder_id}, '{query_text}')",
                }

            return weakand_match_query_fn

        def make_hybrid_match_fn(embedding_field, query_tensor, embedder_id):
            def hybrid_match_query_fn(query_text: str, top_k: int) -> dict:
                return {
                    "yql": str(
                        qb.select("*")
                        .from_(schema_name)
                        .where(
                            qb.nearestNeighbor(
                                field=embedding_field,
                                query_vector=query_tensor,
                                annotations={"targetHits": 100},
                            )
                            | qb.userQuery(query_text)
                        )
                    ),
                    "query": query_text,
                    "ranking": "match-only",
                    f"input.query({query_tensor})": f"embed({embedder_id}, '{query_text}')",
                }

            return hybrid_match_query_fn

        # Ranking strategies (for VespaEvaluator)
        def make_semantic_fn(
            embedding_field, query_tensor, profile_suffix, embedder_id
        ):
            def semantic_query_fn(query_text: str, top_k: int) -> dict:
                return {
                    "yql": str(
                        qb.select("*")
                        .from_(schema_name)
                        .where(
                            qb.nearestNeighbor(
                                field=embedding_field,
                                query_vector=query_tensor,
                                annotations={"targetHits": 100},
                            )
                        )
                    ),
                    "query": query_text,
                    "ranking": f"semantic{profile_suffix}",
                    f"input.query({query_tensor})": f"embed({embedder_id}, '{query_text}')",
                    "hits": top_k,
                }

            return semantic_query_fn

        def make_bm25_fn(profile_suffix):
            def bm25_query_fn(query_text: str, top_k: int) -> dict:
                return {
                    "yql": "select * from sources * where userQuery();",
                    "query": query_text,
                    "ranking": f"bm25{profile_suffix}",
                    "hits": top_k,
                }

            return bm25_query_fn

        def make_fusion_fn(embedding_field, query_tensor, profile_suffix, embedder_id):
            def fusion_query_fn(query_text: str, top_k: int) -> dict:
                return {
                    "yql": str(
                        qb.select("*")
                        .from_(schema_name)
                        .where(
                            qb.nearestNeighbor(
                                field=embedding_field,
                                query_vector=query_tensor,
                                annotations={"targetHits": 100},
                            )
                            | qb.userQuery(query_text)
                        )
                    ),
                    "query": query_text,
                    "ranking": f"fusion{profile_suffix}",
                    f"input.query({query_tensor})": f"embed({embedder_id}, '{query_text}')",
                    "hits": top_k,
                }

            return fusion_query_fn

        def make_atan_norm_fn(
            embedding_field, query_tensor, profile_suffix, embedder_id
        ):
            def atan_norm_query_fn(query_text: str, top_k: int) -> dict:
                return {
                    "yql": str(
                        qb.select("*")
                        .from_(schema_name)
                        .where(
                            qb.nearestNeighbor(
                                field=embedding_field,
                                query_vector=query_tensor,
                                annotations={"targetHits": 100},
                            )
                            | qb.userQuery(query_text)
                        )
                    ),
                    "query": query_text,
                    "ranking": f"atan_norm{profile_suffix}",
                    f"input.query({query_tensor})": f"embed({embedder_id}, '{query_text}')",
                    "hits": top_k,
                }

            return atan_norm_query_fn

        # Add match strategies
        query_functions[f"match_semantic{model_label}"] = make_semantic_match_fn(
            embedding_field, query_tensor, config.component_id
        )
        query_functions[f"match_hybrid{model_label}"] = make_hybrid_match_fn(
            embedding_field, query_tensor, config.component_id
        )

        # Add ranking strategies
        query_functions[f"semantic{model_label}"] = make_semantic_fn(
            embedding_field, query_tensor, profile_suffix, config.component_id
        )
        query_functions[f"bm25{model_label}"] = make_bm25_fn(profile_suffix)
        query_functions[f"fusion{model_label}"] = make_fusion_fn(
            embedding_field, query_tensor, profile_suffix, config.component_id
        )
        query_functions[f"atan_norm{model_label}"] = make_atan_norm_fn(
            embedding_field, query_tensor, profile_suffix, config.component_id
        )

    # Add weakand match strategy (only once, not model-specific)
    # Use the first model's embedder_id for consistency
    first_embedder_id = model_configs[0].component_id

    def weakand_match_query_fn(query_text: str, top_k: int) -> dict:
        return {
            "yql": str(
                qb.select("*").from_(schema_name).where(qb.userQuery(query_text))
            ),
            "query": query_text,
            "ranking": "match-only",
            "input.query(q)": f"embed({first_embedder_id}, '{query_text}')",
        }

    query_functions["match_weakand"] = weakand_match_query_fn

    return query_functions


def run_match_evaluation(
    app: Vespa, queries: dict, relevant_docs: dict, query_functions: dict
):
    """
    Run match evaluation (VespaMatchEvaluator) for retrieval strategies.

    Args:
        app: Vespa application instance
        queries: Dictionary mapping query IDs to query text
        relevant_docs: Dictionary mapping query IDs to relevant document IDs
        query_functions: Dictionary mapping strategy names to query functions

    Returns:
        DataFrame with match evaluation results
    """
    print("\n" + "=" * 80)
    print("RUNNING MATCH EVALUATION (Retrieval Phase)")
    print("=" * 80)

    match_results = {}
    match_strategies = [k for k in query_functions.keys() if k.startswith("match_")]

    for strategy_name in match_strategies:
        print(f"\nEvaluating {strategy_name}...")
        query_fn = query_functions[strategy_name]

        match_evaluator = VespaMatchEvaluator(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=query_fn,
            app=app,
            name=strategy_name,
            id_field="id",
            write_csv=True,
            write_verbose=True,
        )

        results = match_evaluator()
        match_results[strategy_name] = results
        print(f"Results for {strategy_name}:")
        print(results)

    return pd.DataFrame(match_results)


def run_ranking_evaluation(
    app: Vespa, queries: dict, relevant_docs: dict, query_functions: dict
):
    """
    Run ranking evaluation (VespaEvaluator) for ranking strategies.

    Args:
        app: Vespa application instance
        queries: Dictionary mapping query IDs to query text
        relevant_docs: Dictionary mapping query IDs to relevant document IDs
        query_functions: Dictionary mapping strategy names to query functions

    Returns:
        DataFrame with ranking evaluation results
    """
    print("\n" + "=" * 80)
    print("RUNNING RANKING EVALUATION (Ranking Phase)")
    print("=" * 80)

    ranking_results = {}
    ranking_strategies = [
        k for k in query_functions.keys() if not k.startswith("match_")
    ]

    for strategy_name in ranking_strategies:
        print(f"\nEvaluating {strategy_name}...")
        query_fn = query_functions[strategy_name]

        evaluator = VespaEvaluator(
            queries=queries,
            relevant_docs=relevant_docs,
            vespa_query_fn=query_fn,
            app=app,
            name=strategy_name,
            write_csv=True,
        )

        results = evaluator.run()
        ranking_results[strategy_name] = results

    return pd.DataFrame(ranking_results)


def save_results(
    match_results_df: pd.DataFrame,
    ranking_results_df: pd.DataFrame,
    output_dir: str = ".",
):
    """
    Save evaluation results and create visualizations.

    Args:
        match_results_df: DataFrame with match evaluation results
        ranking_results_df: DataFrame with ranking evaluation results
        output_dir: Directory to save results (default: current directory)
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save match results
    match_csv_path = os.path.join(output_dir, "nanobeir_match_results.csv")
    match_results_df.to_csv(match_csv_path)
    print(f"\nMatch results saved to: {match_csv_path}")
    print("\nMatch Results Summary:")
    print(match_results_df)

    # Save and process ranking results
    ranking_csv_path = os.path.join(output_dir, "nanobeir_ranking_results.csv")
    ranking_results_df.to_csv(ranking_csv_path)
    print(f"\nRanking results saved to: {ranking_csv_path}")

    # Separate searchtime from other metrics
    searchtime = ranking_results_df[ranking_results_df.index.str.contains("searchtime")]
    metrics = ranking_results_df[~ranking_results_df.index.str.contains("searchtime")]

    # Save separate CSVs
    metrics_csv_path = os.path.join(output_dir, "nanobeir_ranking_metrics.csv")
    searchtime_csv_path = os.path.join(output_dir, "nanobeir_searchtime.csv")
    metrics.to_csv(metrics_csv_path)
    searchtime.to_csv(searchtime_csv_path)
    print(f"Ranking metrics saved to: {metrics_csv_path}")
    print(f"Search time saved to: {searchtime_csv_path}")

    print("\nRanking Metrics Summary:")
    print(metrics)

    print("\nSearch Time Summary (ms):")
    print(searchtime * 1000)

    # Try to create visualizations if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Plot ranking metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics.plot(kind="bar", ax=ax)
        ax.set_title("NanoBEIR Ranking Metrics Comparison")
        ax.set_ylabel("Score")
        plt.tight_layout()
        metrics_plot_path = os.path.join(output_dir, "nanobeir_ranking_metrics.png")
        plt.savefig(metrics_plot_path)
        print(f"\nRanking metrics plot saved to: {metrics_plot_path}")
        plt.close()

        # Plot search time
        fig, ax = plt.subplots(figsize=(12, 6))
        (searchtime * 1000).plot(kind="bar", ax=ax)
        ax.set_title("NanoBEIR Search Time Comparison")
        ax.set_ylabel("Time (ms)")
        plt.tight_layout()
        searchtime_plot_path = os.path.join(output_dir, "nanobeir_searchtime.png")
        plt.savefig(searchtime_plot_path)
        print(f"Search time plot saved to: {searchtime_plot_path}")
        plt.close()

    except ImportError:
        print("\nNote: matplotlib not available, skipping plot generation")


def main():
    """
    Main function to run the complete NanoBEIR evaluation workflow.
    """
    print("=" * 80)
    print("NanoBEIR EVALUATION RUNNER")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Tenant: {TENANT_NAME}")
    print(f"  Application: {APPLICATION}")
    print(f"  Schema: {SCHEMA_NAME}")
    print(f"  Dataset: {DATASET_ID}")
    print(f"  Models: {MODELS}")

    # Create application package
    print("\n" + "=" * 80)
    print("CREATING APPLICATION PACKAGE")
    print("=" * 80)
    package = create_evaluation_package(
        MODELS,
        app_name=APPLICATION,
        schema_name=SCHEMA_NAME,
    )
    package.to_files("evaltest")
    print("\nCreated package with:")
    print(f"  - {len(package.components)} embedding component(s)")
    print(f"  - {len(package.schema.rank_profiles)} rank profile(s)")
    embedding_fields = [
        f for f in package.schema.document.fields if f.name.startswith("embedding")
    ]
    print(f"  - {len(embedding_fields)} embedding field(s)")

    if TARGET == DeployTarget.VESPA_CLOUD:
        # Deploy to Vespa Cloud
        print("\n" + "=" * 80)
        print("DEPLOYING TO VESPA CLOUD")
        print("=" * 80)
        vespa_cloud = VespaCloud(
            tenant=TENANT_NAME,
            application=APPLICATION,
            key_content=os.getenv("VESPA_TEAM_API_KEY", None),
            application_package=package,
        )
        app: Vespa = vespa_cloud.deploy(max_wait=1800)
    elif TARGET == DeployTarget.LOCAL:
        # Deploy locally using Docker
        print("\n" + "=" * 80)
        print("DEPLOYING LOCALLY WITH DOCKER")
        print("=" * 80)
        vespa_docker = VespaDocker()
        app: Vespa = vespa_docker.deploy(
            application_package=package,
        )
    print("Deployment successful!")

    try:
        # Feed data
        print("\n" + "=" * 80)
        print("FEEDING DATA")
        print("=" * 80)
        feed_data(app, DATASET_ID, SCHEMA_NAME)

        # Load queries and qrels
        queries, relevant_docs = load_queries_and_qrels(DATASET_ID)

        # Get model configs for query function creation
        model_configs = [
            get_model_config(m) if isinstance(m, str) else m for m in MODELS
        ]

        # Create query functions
        print("\n" + "=" * 80)
        print("CREATING QUERY FUNCTIONS")
        print("=" * 80)
        query_functions = create_query_functions(model_configs, SCHEMA_NAME)
        print(f"Created {len(query_functions)} query functions:")
        for name in query_functions.keys():
            print(f"  - {name}")

        # Run match evaluation
        match_results = run_match_evaluation(
            app, queries, relevant_docs, query_functions
        )

        # Run ranking evaluation
        ranking_results = run_ranking_evaluation(
            app, queries, relevant_docs, query_functions
        )

        # Save results
        save_results(match_results, ranking_results)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR DURING EVALUATION")
        print("=" * 80)
        print(f"Exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Clean up
        print("\n" + "=" * 80)
        print("CLEANING UP")
        print("=" * 80)
        print("Deleting Vespa application...")
        # vespa_cloud.delete()
        print("Cleanup complete!")


if __name__ == "__main__":
    main()
