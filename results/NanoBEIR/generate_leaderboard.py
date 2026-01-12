# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///
"""This script processes multiple NanoBEIR benchmark result JSON files and generates a consolidated
JavaScript file (models.js) containing model performance metrics for leaderboard display, that is
intended to be used for a web-based frontend.
"""

import json
import glob
import re
import numpy as np
from collections import defaultdict

# --- Configuration ---
# Path to your JSON result files (e.g., "results/*.json")
INPUT_FILES_PATTERN = "*.json"
# Path to benchmark result files
BENCHMARK_FILES_PATTERN = "../inference/benchmark_results/*.json"

# Expected NanoBEIR tasks
EXPECTED_TASKS = [
    "NanoArguAnaRetrieval",
    "NanoClimateFeverRetrieval",
    "NanoDBPediaRetrieval",
    "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoHotpotQARetrieval",
    "NanoMSMARCORetrieval",
    "NanoNFCorpusRetrieval",
    "NanoNQRetrieval",
    "NanoQuoraRetrieval",
    "NanoSCIDOCSRetrieval",
    "NanoSciFactRetrieval",
    "NanoTouche2020Retrieval",
]


# --- Helpers ---
def get_org_and_name(model_id):
    """Extracts nice Organization and Name from a HuggingFace ID."""
    # Heuristic: convert snake_case to Kebab-Case or standard text
    parts = model_id.replace("_", "-").split("/")

    if len(parts) > 1:
        org = parts[0]
        name = parts[1]
    else:
        # Fallback if no Org present
        org = "Community"
        name = parts[0]

    # Clean up name for display
    clean_name = name.replace("-", " ").title()
    clean_name = (
        clean_name.replace("Bert", "BERT").replace("Gte", "GTE").replace("E5", "E5")
    )

    return org, clean_name, model_id


def parse_result_key(key, configs=None):
    """
    Parses keys like: 'semantic_lightonai_modernbert_large_128_int8' (old format)
    or just 'semantic', 'bm25', etc. (new format)

    For new format, uses configs to determine dim and dtype.
    Returns: (query_func, dimension, dtype) or None
    """
    # Valid query functions we want to track
    valid_funcs = ["semantic", "bm25", "fusion", "atan_norm", "norm_linear"]

    # Check for new simple format (key is just the query function)
    if key in valid_funcs:
        # Use the first config to get dimension and dtype
        if configs and len(configs) > 0:
            config = configs[0]
            dim = config.get("embedding_dim", 768)
            # Determine dtype from config
            if config.get("binarized") or config.get("embedding_field_type") == "int8":
                dtype = "int8"
            elif config.get("embedding_field_type") == "bfloat16":
                dtype = "bfloat16"
            else:
                dtype = "float"
            return key, dim, dtype
        return None

    # Old format: Match known prefixes explicitly
    prefix = None
    for func in valid_funcs:
        if key.startswith(func + "_"):
            prefix = func
            break

    if not prefix:
        return None

    # Remove prefix and parse the rest: model_dim_dtype
    rest = key[len(prefix) + 1 :]  # +1 for the underscore

    # The last two parts are dim and dtype, separated by underscore
    # e.g., "e5_small_v2_384_float" -> dim=384, dtype=float
    match = re.match(r"(.+)_(\d+)_([a-z0-9]+)$", rest)
    if match:
        dim = int(match.group(2))
        dtype = match.group(3)
        return prefix, dim, dtype

    return None


def load_benchmark_data(file_pattern):
    """
    Loads benchmark data from JSON files.
    Returns a dict: {model_id: {hardware_type: { ... data ... }}}
    """
    benchmarks = defaultdict(dict)
    files = glob.glob(file_pattern)
    print(f"Found {len(files)} benchmark files. Processing...")

    for file_path in files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Benchmark files are expected to be a list of objects
            if not isinstance(data, list):
                print(
                    f"  INFO [{file_path}]: Not a list, skipping (might be a NanoBEIR result file)"
                )
                continue

            for entry in data:
                model_id = entry.get("model_id")
                hw_type = entry.get("hardware_type")
                if model_id and hw_type:
                    benchmarks[model_id][hw_type] = entry

        except Exception as e:
            print(f"  ERROR [{file_path}]: {e}")

    return benchmarks


def process_benchmark_data(file_pattern, benchmark_pattern):
    # Load benchmarks first
    benchmark_data = load_benchmark_data(benchmark_pattern)

    # Data Structure:
    # models[model_id][variant_key] = [list of scores from different tasks]
    # variant_key format: "{query_func}_{dim}_{dtype}" e.g. "semantic_384_float"
    models_data = defaultdict(lambda: defaultdict(list))

    files = glob.glob(file_pattern)
    print(f"Found {len(files)} files. Processing...")

    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Skip if it looks like a benchmark file (list)
        if isinstance(data, list):
            continue

        # Check for missing tasks
        results = data.get("results", {})
        found_tasks = set(results.keys())
        expected_tasks = set(EXPECTED_TASKS)
        missing_tasks = expected_tasks - found_tasks
        extra_tasks = found_tasks - expected_tasks

        if missing_tasks:
            print(
                f"  WARNING [{file_path}]: Missing {len(missing_tasks)} tasks: {sorted(missing_tasks)}"
            )
        if extra_tasks:
            print(
                f"  INFO [{file_path}]: Found {len(extra_tasks)} unexpected tasks: {sorted(extra_tasks)}"
            )

        # 1. Parse Metadata to determine model grouping
        configs = data.get("metadata", {}).get("model_configs", [])

        # Get the primary model_id from the first config
        if not configs:
            print(f"  WARNING [{file_path}]: No model_configs found, skipping")
            continue
        primary_model_id = configs[0].get("model_id", "unknown")
        max_tokens = configs[0].get("max_tokens")
        print(f"  Processing model: {primary_model_id}")

        for task_name, task_results in results.items():
            for key, score_data in task_results.items():
                # Check if valid score exists
                if not score_data or "scores" not in score_data:
                    continue

                # Handle potential splits (train/test/dev)
                # Note: score_data["scores"] can be None, so we need to handle that
                subset = score_data.get("scores")
                if not subset:
                    continue
                split_scores = (
                    subset.get("test")
                    or subset.get("train")
                    or subset.get("validation")
                )

                if not split_scores:
                    continue

                ndcg_score = split_scores[0].get("ndcg_at_10", 0)

                # Parse the key to classify the score (pass configs for new simple format)
                parsed = parse_result_key(key, configs)
                if not parsed:
                    continue

                query_func, dim, dtype = parsed

                # For int8/binary in old format, the dimension in the key is the packed dimension (original_dim / 8)
                # For new format, dim is already the original dimension from config
                original_dim = dim
                # Only apply the *8 conversion for old format (where dim is packed)
                if dtype == "int8" and key not in [
                    "semantic",
                    "bm25",
                    "fusion",
                    "atan_norm",
                    "norm_linear",
                ]:
                    original_dim = dim * 8  # Convert packed dimension back to original

                # Use the primary model_id from metadata (normalized with dashes)
                raw_model_id = primary_model_id.replace("_", "-")

                # Categorize score by query_func, original_dim, dtype
                # Use "{query_func}_{original_dim}_{dtype}" as key for consistent dimension tracking
                variant_key = f"{query_func}_{original_dim}_{dtype}"
                models_data[raw_model_id][variant_key].append(ndcg_score)

                # Track metadata
                if dtype == "int8":
                    models_data[raw_model_id]["_binary_supported"] = True
                if dtype == "bfloat16":
                    models_data[raw_model_id]["_bfloat16_supported"] = True
                if max_tokens is not None:
                    models_data[raw_model_id]["_max_tokens"] = max_tokens

    # --- Formatting Output ---
    output_list = []

    # Collect BM25 scores from all models (should be the same)
    bm25_scores = []

    for model_id, variants in models_data.items():
        # Collect all dimensions by dtype
        dims_by_dtype = defaultdict(set)
        for k in variants.keys():
            if k.startswith("_"):
                continue
            parts = k.split("_")
            # Format: query_func_dim_dtype e.g. "semantic_384_float"
            if len(parts) == 3:
                try:
                    dim = int(parts[1])
                    dtype = parts[2]
                    dims_by_dtype[dtype].add(dim)
                except ValueError:
                    continue

        # Find max dimension (prefer float, then bfloat16, then int8)
        all_dims = (
            dims_by_dtype.get("float", set())
            or dims_by_dtype.get("bfloat16", set())
            or dims_by_dtype.get("int8", set())
        )
        if not all_dims:
            continue

        max_dim = max(all_dims)
        all_float_dims = sorted(dims_by_dtype.get("float", set()), reverse=True)
        all_bfloat16_dims = sorted(dims_by_dtype.get("bfloat16", set()), reverse=True)
        all_int8_dims = sorted(dims_by_dtype.get("int8", set()), reverse=True)

        # Helper to get average score for a variant key
        def get_score(key):
            if key in variants and variants[key]:
                return round(np.mean(variants[key]), 3)
            return None

        # Collect BM25 scores (should be same across all models)
        for k, v in variants.items():
            if k.startswith("bm25_"):
                bm25_scores.extend(v)

        # Build scores object - all query functions for each dim/dtype combination
        scores_obj = {}

        # Query functions: semantic (vector-only), and hybrid methods
        query_funcs = ["semantic", "fusion", "atan_norm", "norm_linear"]

        # For each dtype, get scores for all dimensions and query functions
        dtype_configs = [
            ("float", "float", all_float_dims),
            ("bfloat16", "bfloat16", all_bfloat16_dims),
            ("int8", "binary", all_int8_dims),
        ]

        for dtype, dtype_key, dims in dtype_configs:
            for dim in dims:
                for qf in query_funcs:
                    key = f"{qf}_{dim}_{dtype}"
                    score = get_score(key)
                    if score is not None:
                        # Include dimension in the score key for MRL support
                        scores_obj[f"{qf}_{dim}_{dtype_key}"] = score

        # Determine MRL support: more than one dimension for float or bfloat16
        mrl_support = len(all_float_dims) > 1 or len(all_bfloat16_dims) > 1

        # Org / Name Parsing
        org, name, hf_id = get_org_and_name(model_id)

        # Retrieve benchmark data for this model
        # model_id here is already sanitized (replaced _ with -)
        model_benchmarks = benchmark_data.get(model_id, {})

        # Try to fill speeds if possible (heuristic)
        speeds = {
            "t4": 0,
            "c7g": 0,
        }
        # Map known hardware types to speeds keys
        hw_map = {
            "c7g.2xlarge": "c7g",
            "t4": "t4",  # Assuming t4 is the key for T4
        }

        for hw_type, data in model_benchmarks.items():
            # Update speeds if we have a mapping
            if hw_type in hw_map:
                speeds[hw_map[hw_type]] = data.get("queries_throughput", 0)

        model_entry = {
            "id": model_id.replace("/", "-").replace("_", "-"),
            "name": name,
            "org": org,
            "modelId": hf_id,
            "params": "TODO",  # Not available in result JSONs
            "maxDim": max_dim,
            "contextLength": variants.get("_max_tokens"),
            "dimensions": {
                "float": all_float_dims,
                "bfloat16": all_bfloat16_dims,
                "binary": all_int8_dims,
            },
            "speeds": speeds,
            "mrlSupport": mrl_support,
            "binarySupport": variants.get("_binary_supported", False),
            "bfloat16Support": variants.get("_bfloat16_supported", False),
            "scores": scores_obj,
            "benchmarks": model_benchmarks,
        }

        output_list.append(model_entry)

    # Add BM25 as a separate "model" entry
    if bm25_scores:
        bm25_entry = {
            "id": "bm25",
            "name": "BM25",
            "org": "Vespa",
            "modelId": "vespa-bm25",
            "params": "N/A",
            "maxDim": None,
            "contextLength": None,
            "speeds": {
                "t4": 0,
                "c7g": 0,
            },
            "mrlSupport": False,
            "binarySupport": False,
            "bfloat16Support": False,
            "isBM25": True,
            "scores": {
                "semantic_float": round(np.mean(bm25_scores), 3),
                "fusion_float": None,
                "semantic_bfloat16": None,
                "fusion_bfloat16": None,
                "semantic_binary": None,
                "fusion_binary": None,
            },
        }
        output_list.append(bm25_entry)

    return output_list


OUTPUT_FILE = "models.js"

if __name__ == "__main__":
    results = process_benchmark_data(INPUT_FILES_PATTERN, BENCHMARK_FILES_PATTERN)

    output_content = f"const models = {json.dumps(results, indent=4)};"

    with open(OUTPUT_FILE, "w") as f:
        f.write(output_content)

    print(f"Generated {OUTPUT_FILE} with {len(results)} models.")
