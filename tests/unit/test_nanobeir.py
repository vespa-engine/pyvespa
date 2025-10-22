"""
Tests for NanoBEIR evaluation utilities.
"""

import pytest
from vespa.nanobeir import (
    ModelConfig,
    create_embedder_component,
    create_embedding_field,
    create_semantic_rank_profile,
    create_hybrid_rank_profile,
    get_model_config,
    COMMON_MODELS,
)
from vespa.package import Component, Field, RankProfile


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_basic_config(self):
        """Test basic model configuration."""
        config = ModelConfig(
            model_id="test-model",
            embedding_dim=384,
        )
        assert config.model_id == "test-model"
        assert config.embedding_dim == 384
        assert config.tokenizer_id == "test-model"  # Defaults to model_id
        assert config.binarized is False
        assert config.component_id == "test_model"  # Hyphens replaced

    def test_config_with_tokenizer(self):
        """Test configuration with separate tokenizer."""
        config = ModelConfig(
            model_id="e5-small-v2",
            embedding_dim=384,
            tokenizer_id="e5-base-v2-vocab",
        )
        assert config.tokenizer_id == "e5-base-v2-vocab"

    def test_config_binarized(self):
        """Test binarized model configuration."""
        config = ModelConfig(
            model_id="bge-m3",
            embedding_dim=1024,
            binarized=True,
        )
        assert config.binarized is True

    def test_config_with_paths(self):
        """Test configuration with local paths."""
        config = ModelConfig(
            model_id="custom-model",
            embedding_dim=512,
            model_path="/path/to/model.onnx",
            tokenizer_path="/path/to/tokenizer.json",
        )
        assert config.model_path == "/path/to/model.onnx"
        assert config.tokenizer_path == "/path/to/tokenizer.json"

    def test_config_invalid_dimension(self):
        """Test that invalid embedding dimension raises error."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            ModelConfig(model_id="test", embedding_dim=0)

        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            ModelConfig(model_id="test", embedding_dim=-1)

    def test_component_id_sanitization(self):
        """Test that component IDs are properly sanitized."""
        config = ModelConfig(
            model_id="some/model-with-special_chars",
            embedding_dim=384,
        )
        # Slashes and hyphens should be replaced with underscores
        assert config.component_id == "some_model_with_special_chars"


class TestCreateEmbedderComponent:
    """Test create_embedder_component function."""

    def test_component_with_model_id(self):
        """Test component creation with model ID."""
        config = ModelConfig(
            model_id="e5-small-v2",
            embedding_dim=384,
            tokenizer_id="e5-base-v2-vocab",
        )
        component = create_embedder_component(config)

        assert isinstance(component, Component)
        assert component.id == "e5_small_v2"
        assert component.type == "hugging-face-embedder"
        assert len(component.parameters) == 2

        # Check transformer-model parameter
        assert component.parameters[0].name == "transformer-model"
        assert component.parameters[0].args == {"model-id": "e5-small-v2"}

        # Check tokenizer-model parameter
        assert component.parameters[1].name == "tokenizer-model"
        assert component.parameters[1].args == {"model-id": "e5-base-v2-vocab"}

    def test_component_with_paths(self):
        """Test component creation with file paths."""
        config = ModelConfig(
            model_id="custom-model",
            embedding_dim=384,
            model_path="/models/custom.onnx",
            tokenizer_path="/models/tokenizer.json",
        )
        component = create_embedder_component(config)

        assert component.parameters[0].args == {"path": "/models/custom.onnx"}
        assert component.parameters[1].args == {"path": "/models/tokenizer.json"}


class TestCreateEmbeddingField:
    """Test create_embedding_field function."""

    def test_float_embedding_field(self):
        """Test field creation for float embeddings."""
        config = ModelConfig(
            model_id="e5-small-v2",
            embedding_dim=384,
            binarized=False,
        )
        field = create_embedding_field(config)

        assert isinstance(field, Field)
        assert field.name == "embedding"
        assert field.type == "tensor<float>(x[384])"
        assert field.is_document_field is False

        # Check indexing statement
        assert "input text" in field.indexing
        assert "embed" in field.indexing
        assert "index" in field.indexing
        assert "attribute" in field.indexing
        assert "pack_bits" not in field.indexing

        # Check HNSW configuration
        assert field.ann is not None
        assert field.ann.distance_metric == "angular"

    def test_binarized_embedding_field(self):
        """Test field creation for binarized embeddings."""
        config = ModelConfig(
            model_id="bge-m3",
            embedding_dim=1024,
            binarized=True,
        )
        field = create_embedding_field(config)

        assert field.name == "embedding"
        # 1024 bits packed into 128 int8 values
        assert field.type == "tensor<int8>(x[128])"

        # Check indexing statement includes pack_bits
        assert "pack_bits" in field.indexing

        # Check HNSW configuration uses hamming distance
        assert field.ann.distance_metric == "hamming"

    def test_custom_field_name(self):
        """Test field creation with custom name."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        field = create_embedding_field(config, field_name="my_embedding")

        assert field.name == "my_embedding"

    def test_custom_distance_metric(self):
        """Test field creation with custom distance metric."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        field = create_embedding_field(config, distance_metric="euclidean")

        assert field.ann.distance_metric == "euclidean"

    def test_custom_indexing(self):
        """Test field creation with custom indexing."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        custom_indexing = ["attribute", "index"]
        field = create_embedding_field(config, indexing=custom_indexing)

        assert field.indexing == custom_indexing


class TestCreateSemanticRankProfile:
    """Test create_semantic_rank_profile function."""

    def test_float_semantic_profile(self):
        """Test semantic profile for float embeddings."""
        config = ModelConfig(
            model_id="e5-small-v2",
            embedding_dim=384,
            binarized=False,
        )
        profile = create_semantic_rank_profile(config)

        assert isinstance(profile, RankProfile)
        assert profile.name == "semantic"
        assert len(profile.inputs) == 1
        assert profile.inputs[0][0] == "query(q)"
        assert profile.inputs[0][1] == "tensor<float>(x[384])"

        # Check functions
        assert len(profile.functions) == 1
        assert profile.functions[0].name == "similarity"
        assert "closeness(field, embedding)" in profile.functions[0].expression

        assert profile.first_phase == "similarity"
        assert "similarity" in profile.match_features

    def test_binarized_semantic_profile(self):
        """Test semantic profile for binarized embeddings."""
        config = ModelConfig(
            model_id="bge-m3",
            embedding_dim=1024,
            binarized=True,
        )
        profile = create_semantic_rank_profile(config)

        # Query tensor should be int8 with packed dimensions
        assert profile.inputs[0][1] == "tensor<int8>(x[128])"

        # Similarity function should handle hamming distance
        similarity_func = profile.functions[0]
        assert "closeness(field, embedding)" in similarity_func.expression
        # Should have transformation for hamming distance
        assert "1/(1 + " in similarity_func.expression

    def test_custom_profile_name(self):
        """Test semantic profile with custom name."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        profile = create_semantic_rank_profile(config, profile_name="my_semantic")

        assert profile.name == "my_semantic"

    def test_custom_embedding_field(self):
        """Test semantic profile with custom embedding field name."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        profile = create_semantic_rank_profile(
            config,
            embedding_field="my_embedding",
        )

        # Check that custom field name is used in expression
        assert "my_embedding" in profile.functions[0].expression

    def test_custom_query_tensor(self):
        """Test semantic profile with custom query tensor name."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        profile = create_semantic_rank_profile(
            config,
            query_tensor="query_embedding",
        )

        assert profile.inputs[0][0] == "query(query_embedding)"


class TestCreateHybridRankProfile:
    """Test create_hybrid_rank_profile function."""

    def test_hybrid_profile_rrf(self):
        """Test hybrid profile with reciprocal rank fusion."""
        config = ModelConfig(
            model_id="e5-small-v2",
            embedding_dim=384,
            binarized=False,
        )
        profile = create_hybrid_rank_profile(config)

        assert isinstance(profile, RankProfile)
        assert profile.name == "fusion"
        assert profile.inherits == "bm25"

        # Check global phase
        assert profile.global_phase is not None
        assert "reciprocal_rank_fusion" in profile.global_phase.expression
        assert "bm25text" in profile.global_phase.expression
        assert profile.global_phase.rerank_count == 1000

        # Check match features includes both
        assert "similarity" in profile.match_features
        assert "bm25text" in profile.match_features

    def test_hybrid_profile_normalize(self):
        """Test hybrid profile with linear normalization."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        profile = create_hybrid_rank_profile(
            config,
            fusion_method="normalize",
        )

        assert "normalize_linear" in profile.global_phase.expression
        assert "bm25text" in profile.global_phase.expression

    def test_hybrid_profile_binarized(self):
        """Test hybrid profile for binarized embeddings."""
        config = ModelConfig(
            model_id="bge-m3",
            embedding_dim=1024,
            binarized=True,
        )
        profile = create_hybrid_rank_profile(config)

        # Query tensor should be int8
        assert profile.inputs[0][1] == "tensor<int8>(x[128])"

        # Similarity function should handle hamming distance
        similarity_func = profile.functions[0]
        assert "1/(1 + " in similarity_func.expression

    def test_hybrid_profile_invalid_fusion(self):
        """Test that invalid fusion method raises error."""
        config = ModelConfig(model_id="test", embedding_dim=384)

        with pytest.raises(ValueError, match="Unknown fusion_method"):
            create_hybrid_rank_profile(config, fusion_method="invalid")

    def test_custom_profile_name(self):
        """Test hybrid profile with custom name."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        profile = create_hybrid_rank_profile(
            config,
            profile_name="my_hybrid",
        )

        assert profile.name == "my_hybrid"

    def test_custom_base_profile(self):
        """Test hybrid profile with custom base profile."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        profile = create_hybrid_rank_profile(
            config,
            base_profile="custom_bm25",
        )

        assert profile.inherits == "custom_bm25"


class TestPredefinedModels:
    """Test predefined model configurations."""

    def test_common_models_exist(self):
        """Test that common models are defined."""
        assert "e5-small-v2" in COMMON_MODELS
        assert "e5-base-v2" in COMMON_MODELS
        assert "snowflake-arctic-embed-xs" in COMMON_MODELS
        assert "bge-m3-binary" in COMMON_MODELS

    def test_e5_small_v2_config(self):
        """Test e5-small-v2 configuration."""
        config = COMMON_MODELS["e5-small-v2"]
        assert config.model_id == "e5-small-v2"
        assert config.embedding_dim == 384
        assert config.tokenizer_id == "e5-base-v2-vocab"
        assert config.binarized is False

    def test_bge_m3_binary_config(self):
        """Test bge-m3-binary configuration."""
        config = COMMON_MODELS["bge-m3-binary"]
        assert config.model_id == "bge-m3"
        assert config.embedding_dim == 1024
        assert config.binarized is True

    def test_get_model_config_success(self):
        """Test getting a predefined model config."""
        config = get_model_config("e5-small-v2")
        assert config.model_id == "e5-small-v2"
        assert config.embedding_dim == 384

    def test_get_model_config_not_found(self):
        """Test that unknown model raises error."""
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_config("nonexistent-model")

        # Error message should list available models
        try:
            get_model_config("nonexistent-model")
        except KeyError as e:
            assert "Available models" in str(e)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_complete_float_setup(self):
        """Test complete setup for float embeddings."""
        config = ModelConfig(
            model_id="e5-small-v2",
            embedding_dim=384,
            tokenizer_id="e5-base-v2-vocab",
        )

        component = create_embedder_component(config)
        field = create_embedding_field(config)
        semantic_profile = create_semantic_rank_profile(config)
        hybrid_profile = create_hybrid_rank_profile(config)

        # Verify all components work together
        assert component.id == config.component_id
        assert field.type == "tensor<float>(x[384])"
        assert field.ann.distance_metric == "angular"
        assert semantic_profile.inputs[0][1] == "tensor<float>(x[384])"
        assert hybrid_profile.inputs[0][1] == "tensor<float>(x[384])"
        assert "closeness(field, embedding)" in semantic_profile.functions[0].expression

    def test_complete_binarized_setup(self):
        """Test complete setup for binarized embeddings."""
        config = ModelConfig(
            model_id="bge-m3",
            embedding_dim=1024,
            binarized=True,
        )

        component = create_embedder_component(config)
        field = create_embedding_field(config)
        semantic_profile = create_semantic_rank_profile(config)
        hybrid_profile = create_hybrid_rank_profile(config)

        # Verify all components work together
        assert component.id == config.component_id
        assert field.type == "tensor<int8>(x[128])"
        assert field.ann.distance_metric == "hamming"
        assert "pack_bits" in field.indexing
        assert semantic_profile.inputs[0][1] == "tensor<int8>(x[128])"
        assert hybrid_profile.inputs[0][1] == "tensor<int8>(x[128])"
        # Hamming distance should be handled specially
        assert "1/(1 + " in semantic_profile.functions[0].expression
