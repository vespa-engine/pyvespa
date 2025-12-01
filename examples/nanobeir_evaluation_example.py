#!/usr/bin/env python3
"""
Example script demonstrating NanoBEIR evaluation with different models.

This script shows how to easily switch between different embedding models
for evaluation, handling differences in embedding dimensions, tokenizers,
and binary vs. float embeddings.
"""

from vespa.nanobeir import (
    ModelConfig,
    get_model_config,
    create_embedder_component,
    create_embedding_field,
    create_evaluation_package,
)


def main():
    """
    Main function demonstrating evaluation setup with different models.
    """
    print("NanoBEIR Evaluation Example")
    print("=" * 60)

    # Example 1: Single model by name (e5-small-v2)
    print("\n1. Single model: e5-small-v2 (float embeddings, 384 dim)")
    print("-" * 60)
    package_e5_small = create_evaluation_package(
        "e5-small-v2",
        app_name="nanobeirsmall",
    )
    config_e5_small = get_model_config("e5-small-v2")
    print(f"   Model: {config_e5_small.model_id}")
    print(f"   Embedding dim: {config_e5_small.embedding_dim}")
    print(f"   Binarized: {config_e5_small.binarized}")
    print(f"   Component ID: {config_e5_small.component_id}")
    embedding_field = package_e5_small.schema.document.fields[2]
    print(f"   Schema embedding field name: {embedding_field.name}")
    print(f"   Schema embedding field type: {embedding_field.type}")
    print(f"   Number of components: {len(package_e5_small.components)}")
    print(f"   Number of rank profiles: {len(package_e5_small.schema.rank_profiles)}")
    profile_names = [
        p.name if hasattr(p, "name") else str(p)
        for p in package_e5_small.schema.rank_profiles
    ]
    print(f"   Rank profile names: {profile_names}")

    # Example 2: Single model with custom config
    print("\n2. Single model with custom config (512 dim)")
    print("-" * 60)
    custom_config = ModelConfig(
        model_id="custom-embedding-model",
        embedding_dim=512,
        tokenizer_id="bert-base-uncased",
        binarized=False,
    )
    package_custom = create_evaluation_package(
        custom_config,
        app_name="nanobeircustom",
    )
    print(f"   Model: {custom_config.model_id}")
    print(f"   Tokenizer: {custom_config.tokenizer_id}")
    print(f"   Embedding dim: {custom_config.embedding_dim}")
    embedding_field = package_custom.schema.document.fields[2]
    print(f"   Schema embedding field name: {embedding_field.name}")
    print(f"   Schema embedding field type: {embedding_field.type}")

    # Example 3: Multiple models (e5-small-v2 and e5-base-v2)
    print("\n3. Multiple models: e5-small-v2 (384 dim) + e5-base-v2 (768 dim)")
    print("-" * 60)
    package_multi = create_evaluation_package(
        ["e5-small-v2", "e5-base-v2"],
        app_name="nanobeirmulti",
    )
    print("   Number of models: 2")
    print(f"   Number of components: {len(package_multi.components)}")
    print(f"   Component IDs: {[c.id for c in package_multi.components]}")
    embedding_fields = [
        f
        for f in package_multi.schema.document.fields
        if f.name.startswith("embedding")
    ]
    print(f"   Number of embedding fields: {len(embedding_fields)}")
    print(f"   Embedding field names: {[f.name for f in embedding_fields]}")
    print(f"   Embedding field types: {[f.type for f in embedding_fields]}")
    print(f"   Number of rank profiles: {len(package_multi.schema.rank_profiles)}")
    profile_names_multi = [
        p.name if hasattr(p, "name") else str(p)
        for p in package_multi.schema.rank_profiles
    ]
    print(f"   Rank profile names: {profile_names_multi}")

    # Example 4: Multiple models with mixed configs (name + custom config)
    print("\n4. Multiple models: e5-small-v2 + custom model (mixed configs)")
    print("-" * 60)
    custom_mixed = ModelConfig(
        model_id="my-custom-embedder",
        embedding_dim=256,
        binarized=False,
    )
    package_mixed = create_evaluation_package(
        ["e5-small-v2", custom_mixed],
        app_name="nanobeirmixed",
    )
    print(f"   Number of components: {len(package_mixed.components)}")
    print(f"   Component IDs: {[c.id for c in package_mixed.components]}")
    embedding_fields_mixed = [
        f
        for f in package_mixed.schema.document.fields
        if f.name.startswith("embedding")
    ]
    print(f"   Embedding field names: {[f.name for f in embedding_fields_mixed]}")
    print(f"   Embedding field types: {[f.type for f in embedding_fields_mixed]}")

    # Example 5: ModernBERT with advanced configuration
    print("\n5. Single model: nomic-ai-modernbert (ModernBERT-based, 768 dim)")
    print("-" * 60)
    config_modernbert = get_model_config("nomic-ai-modernbert")
    package_modernbert = create_evaluation_package(
        "nomic-ai-modernbert",
        app_name="nanobeirmodern",
    )
    print(f"   Model: {config_modernbert.model_id}")
    print(f"   Embedding dim: {config_modernbert.embedding_dim}")
    print(f"   Max tokens: {config_modernbert.max_tokens}")
    print(f"   Transformer output: {config_modernbert.transformer_output}")
    print(f"   Query prepend: {config_modernbert.query_prepend}")
    print(f"   Document prepend: {config_modernbert.document_prepend}")
    embedding_field = package_modernbert.schema.document.fields[2]
    print(f"   Schema embedding field name: {embedding_field.name}")
    print(f"   Schema embedding field type: {embedding_field.type}")
    print(f"   Distance metric: {embedding_field.ann.distance_metric}")

    # Example 6: List all available predefined models
    print("\n6. Available predefined models:")
    print("-" * 60)
    from vespa.nanobeir import COMMON_MODELS

    for model_name, config in COMMON_MODELS.items():
        binary_str = " (binary)" if config.binarized else ""
        print(f"   - {model_name}: {config.embedding_dim} dim{binary_str}")

    # Example 7: Advanced configuration with URL-based models
    print("\n7. Advanced configuration: URL-based model with custom parameters")
    print("-" * 60)
    gte_config = ModelConfig(
        model_id="gte-multilingual-base",
        embedding_dim=768,
        component_id="gte_multilingual",
        model_url="https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/onnx/model_quantized.onnx",
        tokenizer_url="https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/tokenizer.json",
        transformer_output="token_embeddings",
        max_tokens=8192,
        query_prepend="Represent this sentence for searching relevant passages: ",
        document_prepend="passage: ",
    )

    embedder = create_embedder_component(gte_config)
    embedding_field = create_embedding_field(gte_config)

    print(f"   Model: {gte_config.model_id}")
    print(f"   Embedding dim: {gte_config.embedding_dim}")
    print(f"   Component ID: {embedder.id}")
    print(f"   Max tokens: {gte_config.max_tokens}")
    print(f"   Transformer output: {gte_config.transformer_output}")
    print(f"   Query prepend: {gte_config.query_prepend[:50]}...")
    print(f"   Document prepend: {gte_config.document_prepend}")
    print(f"   Number of parameters: {len(embedder.parameters)}")
    print(f"   Schema embedding field type: {embedding_field.type}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nNext steps:")
    print("1. Deploy the package to Vespa Cloud or local Docker")
    print("2. Load NanoBEIR dataset and feed documents")
    print("3. Run evaluation using VespaEvaluator or VespaMatchEvaluator")
    print("4. Compare results across different models")
    print("\nAdvanced features demonstrated:")
    print("- Using predefined model configurations")
    print("- Creating custom model configurations")
    print("- Single model setup with simple function call")
    print("- Multiple model setup with automatic field/component naming")
    print("- Mixed model configurations (predefined + custom)")
    print("- Binary vs. float embeddings")
    print("- URL-based model loading")
    print("- Additional embedder parameters (transformer-output, max-tokens, prepend)")
    print("\nKey benefits of multi-model support:")
    print("- Evaluate multiple models in single deployment")
    print("- Compare model performance side-by-side")
    print("- Automatic conflict resolution (fields/components named uniquely)")
    print("- Each model gets its own set of rank profiles")


if __name__ == "__main__":
    main()
