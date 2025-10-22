#!/usr/bin/env python3
"""
Example script demonstrating NanoBEIR evaluation with different models.

This script shows how to easily switch between different embedding models
for evaluation, handling differences in embedding dimensions, tokenizers,
and binary vs. float embeddings.
"""

from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    RankProfile,
    Function,
    FieldSet,
)
from vespa.nanobeir import (
    ModelConfig,
    get_model_config,
    create_embedder_component,
    create_embedding_field,
    create_semantic_rank_profile,
    create_hybrid_rank_profile,
)


def create_evaluation_package(
    model_config: ModelConfig,
    app_name: str = "nanobeir_eval",
    schema_name: str = "doc",
) -> ApplicationPackage:
    """
    Create a Vespa application package configured for NanoBEIR evaluation.

    Args:
        model_config: ModelConfig instance defining the embedding model
        app_name: Name of the application (default: "nanobeir_eval")
        schema_name: Name of the schema (default: "doc")

    Returns:
        ApplicationPackage: Configured Vespa application package
    """
    # Create the embedder component
    embedder = create_embedder_component(model_config)

    # Create the embedding field with correct type and indexing
    embedding_field = create_embedding_field(model_config)

    # Create base BM25 rank profile
    bm25_profile = RankProfile(
        name="bm25",
        inputs=[("query(q)", embedding_field.type)],
        functions=[Function(name="bm25text", expression="bm25(text)")],
        first_phase="bm25text",
        match_features=["bm25text"],
    )

    # Create semantic search rank profile
    semantic_profile = create_semantic_rank_profile(model_config)

    # Create hybrid rank profiles
    fusion_profile = create_hybrid_rank_profile(
        model_config,
        profile_name="fusion",
        fusion_method="rrf",
    )

    atan_norm_profile = create_hybrid_rank_profile(
        model_config,
        profile_name="atan_norm",
        fusion_method="normalize",
    )

    # Build the schema
    schema = Schema(
        name=schema_name,
        document=Document(
            fields=[
                Field(
                    name="id",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="text",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                    bolding=True,
                ),
                embedding_field,
            ]
        ),
        fieldsets=[FieldSet(name="default", fields=["text"])],
        rank_profiles=[
            RankProfile(
                name="match-only",
                inputs=[("query(q)", embedding_field.type)],
                first_phase="random",
            ),
            bm25_profile,
            semantic_profile,
            fusion_profile,
            atan_norm_profile,
        ],
    )

    # Create the application package
    package = ApplicationPackage(
        name=app_name,
        schema=[schema],
        components=[embedder],
    )

    return package


def main():
    """
    Main function demonstrating evaluation setup with different models.
    """
    print("NanoBEIR Evaluation Example")
    print("=" * 60)

    # Example 1: E5-small-v2 with float embeddings
    print("\n1. Creating package for e5-small-v2 (float embeddings, 384 dim)")
    print("-" * 60)
    config_e5_small = get_model_config("e5-small-v2")
    package_e5_small = create_evaluation_package(
        config_e5_small,
        app_name="nanobeirsmall",
    )
    print(f"   Model: {config_e5_small.model_id}")
    print(f"   Embedding dim: {config_e5_small.embedding_dim}")
    print(f"   Binarized: {config_e5_small.binarized}")
    print(f"   Component ID: {config_e5_small.component_id}")
    embedding_field = package_e5_small.schema.document.fields[2]
    print(f"   Schema embedding field type: {embedding_field.type}")
    print(f"   Number of rank profiles: {len(package_e5_small.schema.rank_profiles)}")

    # Example 2: E5-base-v2 with larger embeddings
    print("\n2. Creating package for e5-base-v2 (float embeddings, 768 dim)")
    print("-" * 60)
    config_e5_base = get_model_config("e5-base-v2")
    package_e5_base = create_evaluation_package(
        config_e5_base,
        app_name="nanobeirbase",
    )
    print(f"   Model: {config_e5_base.model_id}")
    print(f"   Embedding dim: {config_e5_base.embedding_dim}")
    print(f"   Binarized: {config_e5_base.binarized}")
    embedding_field = package_e5_base.schema.document.fields[2]
    print(f"   Schema embedding field type: {embedding_field.type}")

    # Example 3: BGE-M3 with binary embeddings
    print("\n3. Creating package for bge-m3-binary (binary embeddings, 1024→128 dim)")
    print("-" * 60)
    config_bge_binary = get_model_config("bge-m3-binary")
    package_bge_binary = create_evaluation_package(
        config_bge_binary,
        app_name="nanobeirbinary",
    )
    print(f"   Model: {config_bge_binary.model_id}")
    print(f"   Embedding dim (before packing): {config_bge_binary.embedding_dim}")
    print(f"   Binarized: {config_bge_binary.binarized}")
    embedding_field = package_bge_binary.schema.document.fields[2]
    print(f"   Schema embedding field type: {embedding_field.type}")
    # Check if pack_bits is in indexing
    print(f"   Uses pack_bits: {'pack_bits' in embedding_field.indexing}")
    print(f"   Distance metric: {embedding_field.ann.distance_metric}")

    # Example 4: Custom model configuration
    print("\n4. Creating package with custom model configuration")
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
    print(f"   Schema embedding field type: {embedding_field.type}")

    # Example 5: List all available predefined models
    print("\n5. Available predefined models:")
    print("-" * 60)
    from vespa.nanobeir import COMMON_MODELS

    for model_name, config in COMMON_MODELS.items():
        binary_str = " (binary)" if config.binarized else ""
        print(f"   - {model_name}: {config.embedding_dim} dim{binary_str}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nNext steps:")
    print("1. Deploy the package to Vespa Cloud or local Docker")
    print("2. Load NanoBEIR dataset and feed documents")
    print("3. Run evaluation using VespaEvaluator or VespaMatchEvaluator")
    print("4. Compare results across different models")


if __name__ == "__main__":
    main()
