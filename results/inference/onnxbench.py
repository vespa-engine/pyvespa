# /// script
# requires-python = ">=3.10,<=3.12"
# dependencies = [
#     "onnxruntime==1.23.2",
#     "transformers",
#     "pandas",
#     "tqdm",
#     "requests",
#     "numpy",
#     "pyvespa @ file:///home/thomas/onnxbench/pyvespa"
# ]
# ///

import argparse
import os
import time
import json
import logging
import random
import shutil
import warnings
import sys
from pathlib import Path
from typing import List, Dict, Any, Set

import numpy as np
import pandas as pd
import requests
import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer
from vespa.models import ModelConfig, HF_MODELS

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress heavy warnings from libraries
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

RESULTS_DIR = Path("benchmark_results")
CACHE_DIR = Path("model_cache")
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

SEED = 42

def get_instance_type() -> str:
    """Detect AWS EC2 instance type via instance metadata service (IMDSv2).
    
    Returns the instance type (e.g., 'c7g.2xlarge') or 'unknown' if not on EC2.
    """
    import urllib.request
    import urllib.error
    
    try:
        # IMDSv2 requires a token first
        token_request = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            method="PUT"
        )
        with urllib.request.urlopen(token_request, timeout=2) as response:
            token = response.read().decode("utf-8")
        
        # Use token to get instance type
        metadata_request = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token}
        )
        with urllib.request.urlopen(metadata_request, timeout=2) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError):
        # Not on EC2 or metadata service unavailable
        return "unknown"

# Benchmark configuration
QUERY_BENCHMARK_DURATION_SEC = 10  # How long to run query inference benchmark
DOC_BENCHMARK_DURATION_SEC = 10    # How long to run document inference benchmark

def get_hf_repo_from_url(url: str) -> str:
    """Extract HuggingFace repo ID from a model URL.
    
    Example: https://huggingface.co/nomic-ai/modernbert-embed-base/resolve/main/onnx/model.onnx
    Returns: nomic-ai/modernbert-embed-base
    """
    import re
    match = re.match(r'https://huggingface\.co/([^/]+/[^/]+)/resolve/', url)
    if match:
        return match.group(1)
    return None

# --- Helper Functions ---

def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # torch is not strictly required as we use onnxruntime, but good practice if mixed
    
def download_file(url: str, dest_path: Path):
    """Download a file with progress bar if it doesn't exist."""
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise

def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)

def get_hf_commit_sha(repo_id: str, branch: str = "main") -> str:
    """Get the commit SHA for a HuggingFace repo.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "nomic-ai/modernbert-embed-base")
        branch: Branch name (default: "main")
    
    Returns:
        Short (12 char) commit SHA or "unknown" if unavailable
    """
    try:
        api_url = f"https://huggingface.co/api/models/{repo_id}/revision/{branch}"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get("sha", "unknown")[:12]  # Return short SHA
    except Exception as e:
        logger.warning(f"Could not fetch commit SHA for {repo_id}: {e}")
        return "unknown"

# --- Sample Data ---

# Sample query (~7 words) and document (~100 words) for consistent benchmarking
SAMPLE_QUERY = "What is the best way to learn programming?"

SAMPLE_DOC = """Learning programming effectively requires a combination of theoretical understanding and practical application. 
Begin by choosing a beginner-friendly language like Python, which has clear syntax and extensive documentation. 
Start with fundamental concepts such as variables, data types, loops, and conditional statements. 
Online platforms offer interactive tutorials that provide immediate feedback on your code. 
Practice regularly by working on small projects that interest you, as this builds problem-solving skills. 
Join coding communities where you can ask questions and learn from experienced developers. 
Reading other people's code helps you understand different approaches and best practices. 
Consider contributing to open source projects once you have basic skills."""

def print_text_stats(text: str, tokenizer, name: str):
    """Print character and token counts for a sample text."""
    char_count = len(text)
    encoding = tokenizer(text, add_special_tokens=False)
    token_count = len(encoding['input_ids'])
    
    logger.info(f"Stats for {name}: chars={char_count}, tokens={token_count}")

# --- Benchmarking ---

def run_inference(session, tokenizer, texts: List[str], max_length: int, duration_sec: float, desc: str):
    """Run ONNX inference for a fixed duration and return latency/throughput stats."""
    latencies = []
    
    # ONNX Runtime Input Name (usually 'input_ids', 'attention_mask')
    input_names = [i.name for i in session.get_inputs()]
    
    # Warmup
    warmup_text = ["This is a warmup sentence."]
    w_inputs = tokenizer(warmup_text, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    ort_inputs = {k: v.astype(np.int64) for k, v in w_inputs.items() if k in input_names}
    for _ in range(3):
        session.run(None, ort_inputs)

    # Run inference for the specified duration, cycling through texts
    start_global = time.perf_counter()
    end_time = start_global + duration_sec
    text_idx = 0
    num_texts = len(texts)
    
    with tqdm(desc=f"{desc} ({duration_sec}s)", unit="samples") as pbar:
        while time.perf_counter() < end_time:
            text = texts[text_idx % num_texts]
            text_idx += 1
            
            # Tokenize
            # Use padding=True (pad to longest in batch, which is just the input itself for batch=1)
            # This avoids padding every input to max_length (e.g., 8192 tokens)
            inputs = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="np")
            
            # Prepare ONNX inputs
            # Cast to int64 as usually required by ONNX Runtime for indices
            ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in input_names}
            
            t0 = time.perf_counter()
            session.run(None, ort_inputs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
            pbar.update(1)

    actual_duration = time.perf_counter() - start_global
    samples_processed = len(latencies)
    
    return {
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_latency_ms": np.mean(latencies),
        "total_duration_sec": actual_duration,
        "samples_processed": samples_processed,
        "throughput_samples_per_sec": samples_processed / actual_duration
    }

def benchmark_config(config: ModelConfig, hardware_type: str = "default"):
    logger.info(f"--- Benchmarking: {config.model_id} ({config.embedding_field_type}) on {hardware_type} ---")
    
    # 1. Extract HuggingFace repo from model URL
    hf_repo = get_hf_repo_from_url(config.model_url)
    if not hf_repo:
        logger.error(f"Could not extract HuggingFace repo from URL: {config.model_url}")
        return None
    
    # 2. Prepare Paths - use model-specific folder to avoid conflicts
    model_cache_dir = CACHE_DIR / config.model_id
    model_cache_dir.mkdir(exist_ok=True)
    
    model_filename = config.model_url.split('/')[-1]
    model_path = model_cache_dir / model_filename

    # 3. Download ONNX model
    download_file(config.model_url, model_path)

    model_size_mb = get_file_size_mb(model_path)
    logger.info(f"Model Size: {model_size_mb:.2f} MB")

    # 4. Load Tokenizer directly from HuggingFace (handles config properly)
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    except Exception as e:
        logger.error(f"Error loading tokenizer from {hf_repo}: {e}")
        return None

    # 4. Load ONNX Model
    # Configure to match Vespa's default ONNX runtime settings
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # onnx-execution-mode: sequential
    sess_options.inter_op_num_threads = 1  # onnx-interop-threads: 1
    sess_options.intra_op_num_threads = 4  # onnx-intraop-threads: 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    try:
        session = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        return None

    # 5. Prepend Prefixes (handle None values from vespa.models.ModelConfig)
    query_prepend = config.query_prepend or ""
    document_prepend = config.document_prepend or ""
    prepped_query = f"{query_prepend}{SAMPLE_QUERY}"
    prepped_doc = f"{document_prepend}{SAMPLE_DOC}"

    # 6. Print text stats (not added to results, just for info)
    print_text_stats(prepped_query, tokenizer, "query")
    print_text_stats(prepped_doc, tokenizer, "doc")

    # 7. Run Benchmarks (time-based) - use single sample repeated
    max_tokens = config.max_tokens or 512  # Default to 512 if not specified
    perf_q = run_inference(session, tokenizer, [prepped_query], max_tokens, QUERY_BENCHMARK_DURATION_SEC, "Inferencing Queries")
    perf_d = run_inference(session, tokenizer, [prepped_doc], max_tokens, DOC_BENCHMARK_DURATION_SEC, "Inferencing Docs")

    # 8. Get model commit SHA for reproducibility
    commit_sha = get_hf_commit_sha(hf_repo)
    
    # 9. Compile Results
    result = {
        "hardware_type": hardware_type,
        "model_id": config.model_id,
        "hf_repo": hf_repo,
        "model_url": config.model_url,
        "commit_sha": commit_sha,
        "model_size_mb": round(model_size_mb, 2),
        "embedding_dim": config.embedding_dim,
        "queries_samples_processed": perf_q['samples_processed'],
        "queries_avg_latency_ms": round(perf_q['avg_latency_ms'], 2),
        "queries_p95_latency_ms": round(perf_q['p95_latency_ms'], 2),
        "queries_throughput": round(perf_q['throughput_samples_per_sec'], 2),
        "docs_samples_processed": perf_d['samples_processed'],
        "docs_avg_latency_ms": round(perf_d['avg_latency_ms'], 2),
        "docs_p95_latency_ms": round(perf_d['p95_latency_ms'], 2),
        "docs_throughput": round(perf_d['throughput_samples_per_sec'], 2),
    }
    
    return result

# --- Main Execution ---

def load_existing_results(json_path: Path) -> tuple[list, Set[str]]:
    """Load existing results from JSON file and return results list and set of model IDs."""
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
            model_ids = {r['model_id'] for r in results}
            return results, model_ids
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load existing results from {json_path}: {e}")
    return [], set()


def main(hardware_type: str = "default", overwrite: bool = False):
    """Run benchmarks with optional hardware type for result organization.
    
    Args:
        hardware_type: Hardware identifier (e.g., 'c7g.2xlarge', 'default')
        overwrite: If True, benchmark all models. If False, skip models already in results.
    """
    set_seed(SEED)
    
    # Use HF_MODELS from vespa.models - filter to only models with model_url
    configs_to_benchmark = [
        (name, config) for name, config in HF_MODELS.items() 
        if config.model_url is not None
    ]
    
    logger.info(f"Found {len(configs_to_benchmark)} models with ONNX URLs: {[name for name, _ in configs_to_benchmark]}")
    logger.info(f"Using fixed sample query ({len(SAMPLE_QUERY.split())} words) and doc (~{len(SAMPLE_DOC.split())} words)")
    
    # Filenames without datetime suffix
    csv_path = RESULTS_DIR / f"benchmark_results_{hardware_type}.csv"
    json_path = RESULTS_DIR / f"benchmark_results_{hardware_type}.json"
    
    # Load existing results if not overwriting
    if overwrite:
        all_results = []
        existing_model_ids: Set[str] = set()
        logger.info("Overwrite mode: will benchmark all models")
    else:
        all_results, existing_model_ids = load_existing_results(json_path)
        if existing_model_ids:
            logger.info(f"Found {len(existing_model_ids)} existing results in {json_path}")
            logger.info(f"Existing models: {sorted(existing_model_ids)}")
        else:
            logger.info("No existing results found, will benchmark all models")
    
    # Filter out already benchmarked models
    if not overwrite and existing_model_ids:
        original_count = len(configs_to_benchmark)
        configs_to_benchmark = [
            (name, config) for name, config in configs_to_benchmark
            if config.model_id not in existing_model_ids
        ]
        skipped_count = original_count - len(configs_to_benchmark)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already benchmarked models")
        if not configs_to_benchmark:
            logger.info("All models already benchmarked. Use --overwrite to re-run.")
            return
    
    logger.info(f"Will benchmark {len(configs_to_benchmark)} models: {[name for name, _ in configs_to_benchmark]}")
    
    def save_results():
        """Save current results to CSV and JSON."""
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(csv_path, index=False)
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Results saved to {csv_path} ({len(all_results)} models)")

    # Iterate and Benchmark
    for model_name, config in configs_to_benchmark:
        try:
            logger.info(f"Starting benchmark for: {model_name}")
            res = benchmark_config(config, hardware_type)
            if res:
                all_results.append(res)
                # Save after each successful benchmark
                save_results()
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up memory
        import gc
        gc.collect()

    # 3. Print Final Summary
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        # Print a clean text table
        cols_to_print = ["hardware_type", "model_size_mb", "queries_avg_latency_ms", "queries_throughput", "docs_throughput"]
        print(df[cols_to_print].to_string(index=False))
        print(f"\nFull results saved to {RESULTS_DIR}")
    else:
        logger.error("No results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ONNX embedding models")
    parser.add_argument(
        "hardware_type",
        nargs="?",
        default=None,
        help="Hardware identifier (e.g., 'c7g.2xlarge'). Auto-detected if not provided."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing results. If False, only benchmark models not in results file."
    )
    args = parser.parse_args()
    
    if args.hardware_type:
        hardware_type = args.hardware_type
    else:
        hardware_type = get_instance_type()
        logger.info(f"Auto-detected hardware type: {hardware_type}")
    
    main(hardware_type=hardware_type, overwrite=args.overwrite)