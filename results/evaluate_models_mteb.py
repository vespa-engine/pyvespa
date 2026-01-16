# Run benchmark evaluation for all HF_MODELS with multiple configurations
import os
from dataclasses import replace
from vespa.models import HF_MODELS, ModelConfig
from vespa.evaluation import VespaMTEBEvaluator

# Deployment configuration via environment variables
# Set VESPA_DEPLOYMENT_TARGET to "cloud" or "docker" (default: "docker" for local testing)
deployment_target = os.getenv("VESPA_DEPLOYMENT_TARGET", "docker")

# Cloud-specific configuration (required when deployment_target="cloud")
vespa_tenant = os.getenv("VESPA_TENANT")
vespa_api_key = os.getenv("VESPA_API_KEY")

if deployment_target == "cloud":
    if vespa_tenant is None:
        raise ValueError(
            "VESPA_TENANT environment variable is required for cloud deployment"
        )
    print(f"Using Vespa Cloud deployment (tenant={vespa_tenant})")
else:
    print("Using Docker deployment")

# Matryoshka models get additional dimension variations
MATRYOSHKA_DIMS = {
    "embeddinggemma-300m": [768, 512, 128],
    "embeddinggemma-300m-q4": [768, 512, 128],
}

# 3 variations: (binarized, embedding_field_type)
VARIATIONS = [
    (True, "int8", "hamming"),  # binary
    (False, "bfloat16", "angular"),  # bfloat16
    (False, "float", "angular"),  # float
]

all_configs: list[ModelConfig] = []

for model_name, base_config in HF_MODELS.items():
    print(f"Creating variations for: {model_name}")

    # Get dimensions to test (just base dim for normal models)
    if model_name in MATRYOSHKA_DIMS:
        dims = sorted(set(MATRYOSHKA_DIMS[model_name]), reverse=True)
    else:
        dims = [base_config.embedding_dim]

    for dim in dims:
        for binarized, field_type, distance_metric in VARIATIONS:
            config = replace(
                base_config,
                embedding_dim=dim,
                binarized=binarized,
                embedding_field_type=field_type,
                distance_metric=distance_metric,
            )
            all_configs.append(config)
            print(f"  - dim={dim}, binarized={binarized}, type={field_type}")

print(f"\nTotal configurations to evaluate: {len(all_configs)}")
print("=" * 60)

# Run evaluations one model at a time
for i, model_config in enumerate(all_configs):
    print(f"\n[{i + 1}/{len(all_configs)}] Evaluating: {model_config.model_id}")
    print(f"  Embedding dim: {model_config.embedding_dim}")
    print(f"  Binarized: {model_config.binarized}")
    print(f"  Field type: {model_config.embedding_field_type}")
    print("-" * 40)

    try:
        if deployment_target == "cloud":
            evaluator = VespaMTEBEvaluator(
                model_configs=model_config,
                benchmark_name="NanoBEIR",
                results_dir="results",
                overwrite=False,
                deployment_target="cloud",
                tenant=vespa_tenant,
                application="mteb-benchmark",
                key_content=vespa_api_key,
                auto_cleanup=True,
            )
        else:
            evaluator = VespaMTEBEvaluator(
                model_configs=model_config,
                benchmark_name="NanoBEIR",
                results_dir="results",
                overwrite=False,
                deployment_target="docker",
                port=8080,
            )
        results = evaluator.evaluate()
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    print(f"  Completed: {model_config.model_id}")

print("\n" + "=" * 60)
print("All evaluations complete!")
