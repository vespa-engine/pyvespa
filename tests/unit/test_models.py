import pytest

from vespa.configuration.vt import compare_xml
from vespa.models import (
    COMMON_MODELS,
    ModelConfig,
    create_embedder_component,
    create_embedding_field,
    create_hybrid_package,
    create_hybrid_rank_profile,
    create_semantic_rank_profile,
    get_model_config,
    list_models,
)
from vespa.package import Component, Field, FirstPhaseRanking, RankProfile


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

    def test_config_with_urls(self):
        """Test configuration with URLs."""
        config = ModelConfig(
            model_id="url-model",
            embedding_dim=768,
            model_url="https://example.com/model.onnx",
            tokenizer_url="https://example.com/tokenizer.json",
        )
        assert config.model_url == "https://example.com/model.onnx"
        assert config.tokenizer_url == "https://example.com/tokenizer.json"

    def test_config_with_explicit_parameters(self):
        """Test configuration with explicit huggingface embedder parameters."""
        config = ModelConfig(
            model_id="custom-model",
            embedding_dim=768,
            max_tokens=8192,
            transformer_output="token_embeddings",
            pooling_strategy="cls",
            normalize=True,
            query_prepend="query: ",
            document_prepend="passage: ",
        )
        assert config.max_tokens == 8192
        assert config.transformer_output == "token_embeddings"
        assert config.pooling_strategy == "cls"
        assert config.normalize is True
        assert config.query_prepend == "query: "
        assert config.document_prepend == "passage: "

    def test_config_pooling_strategy_validation(self):
        """Test that invalid pooling strategy raises error."""
        with pytest.raises(ValueError, match="pooling_strategy must be one of"):
            ModelConfig(
                model_id="test",
                embedding_dim=384,
                pooling_strategy="invalid",
            )

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

    def test_binarized_dimension_validation(self):
        """Test that binarized embeddings require dimension divisible by 8."""
        # Valid binarized dimensions (divisible by 8)
        config = ModelConfig(model_id="test", embedding_dim=1024, binarized=True)
        assert config.embedding_dim == 1024

        config = ModelConfig(model_id="test", embedding_dim=768, binarized=True)
        assert config.embedding_dim == 768

        # Invalid binarized dimensions (not divisible by 8)
        with pytest.raises(
            ValueError,
            match="binarized embeddings require embedding_dim divisible by 8",
        ):
            ModelConfig(model_id="test", embedding_dim=1023, binarized=True)

        with pytest.raises(
            ValueError,
            match="binarized embeddings require embedding_dim divisible by 8",
        ):
            ModelConfig(model_id="test", embedding_dim=385, binarized=True)

        # Non-binarized embeddings can have any positive dimension
        config = ModelConfig(model_id="test", embedding_dim=385, binarized=False)
        assert config.embedding_dim == 385


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
        assert len(component.parameters) == 1

        # Check transformer-model parameter
        assert component.parameters[0].name == "transformer-model"
        assert component.parameters[0].args == {"model-id": "e5-small-v2"}

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

    def test_component_with_urls(self):
        """Test component creation with URLs."""
        config = ModelConfig(
            model_id="url-model",
            embedding_dim=768,
            model_url="https://huggingface.co/model.onnx",
            tokenizer_url="https://huggingface.co/tokenizer.json",
        )
        component = create_embedder_component(config)

        assert component.parameters[0].args == {
            "url": "https://huggingface.co/model.onnx"
        }
        assert component.parameters[1].args == {
            "url": "https://huggingface.co/tokenizer.json"
        }

    def test_component_with_explicit_parameters(self):
        """Test component creation with explicit huggingface embedder parameters."""
        config = ModelConfig(
            model_id="advanced-model",
            embedding_dim=768,
            max_tokens=8192,
            transformer_output="token_embeddings",
            pooling_strategy="cls",
            normalize=True,
        )
        component = create_embedder_component(config)

        # Should have transformer-model, plus 4 explicit parameters
        assert len(component.parameters) == 5
        assert component.parameters[1].name == "max-tokens"
        assert component.parameters[1].children == "8192"
        assert component.parameters[2].name == "transformer-output"
        assert component.parameters[2].children == "token_embeddings"
        assert component.parameters[3].name == "pooling-strategy"
        assert component.parameters[3].children == "cls"
        assert component.parameters[4].name == "normalize"
        assert component.parameters[4].children == "true"

    def test_component_with_prepend_parameters(self):
        """Test component creation with prepend parameters."""
        config = ModelConfig(
            model_id="prepend-model",
            embedding_dim=768,
            model_url="https://example.com/model.onnx",
            tokenizer_url="https://example.com/tokenizer.json",
            query_prepend="Represent this sentence for searching relevant passages: ",
            document_prepend="passage: ",
        )
        component = create_embedder_component(config)

        # Should have transformer-model, tokenizer-url plus prepend parameter
        assert len(component.parameters) == 3
        prepend_param = component.parameters[2]
        assert prepend_param.name == "prepend"
        assert isinstance(prepend_param.children, list)
        assert len(prepend_param.children) == 2
        assert prepend_param.children[0].name == "query"
        assert (
            prepend_param.children[0].children
            == "Represent this sentence for searching relevant passages: "
        )
        assert prepend_param.children[1].name == "document"
        assert prepend_param.children[1].children == "passage: "

    def test_component_with_only_query_prepend(self):
        """Test component creation with only query prepend."""
        config = ModelConfig(
            model_id="query-prepend-model",
            embedding_dim=768,
            query_prepend="query: ",
        )
        component = create_embedder_component(config)

        # Should have transformer-model plus prepend parameter
        assert len(component.parameters) == 2
        prepend_param = component.parameters[1]
        assert prepend_param.name == "prepend"
        assert len(prepend_param.children) == 1
        assert prepend_param.children[0].name == "query"

    def test_component_url_priority_over_path(self):
        """Test that URL takes priority over path when both are provided."""
        config = ModelConfig(
            model_id="test-model",
            embedding_dim=384,
            model_path="/path/to/model.onnx",
            model_url="https://example.com/model.onnx",
            tokenizer_path="/path/to/tokenizer.json",
            tokenizer_url="https://example.com/tokenizer.json",
        )
        component = create_embedder_component(config)

        # URLs should take priority
        assert component.parameters[0].args == {"url": "https://example.com/model.onnx"}
        assert component.parameters[1].args == {
            "url": "https://example.com/tokenizer.json"
        }


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

        # Check indexing statement includes embedder ID
        assert "input text" in field.indexing
        assert "embed e5_small_v2" in field.indexing
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

        # Check indexing statement includes pack_bits and embedder ID
        assert "embed bge_m3" in field.indexing
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

    def test_custom_embedder_id(self):
        """Test field creation with custom embedder ID."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        field = create_embedding_field(config, embedder_id="my_embedder")

        assert "embed my_embedder" in field.indexing


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

    def test_hybrid_profile_atan_norm(self):
        """Test hybrid profile with atan normalization."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        profile = create_hybrid_rank_profile(
            config,
            fusion_method="atan_norm",
        )

        # atan_norm should not have global phase
        assert profile.global_phase is None
        # first_phase should be a FirstPhaseRanking object
        assert isinstance(profile.first_phase, FirstPhaseRanking)
        # Should use atan-normalized sum in first phase
        assert "normalized_bm25 + cos_sim" in profile.first_phase.expression
        # Should have scale, normalized_bm25, and cos_sim functions
        func_names = [f.name for f in profile.functions]
        assert "scale" in func_names
        assert "normalized_bm25" in func_names
        assert "cos_sim" in func_names
        # Check match features
        assert "cos_sim" in profile.match_features
        assert "normalized_bm25" in profile.match_features

    def test_hybrid_profile_norm_linear(self):
        """Test hybrid profile with linear normalization."""
        config = ModelConfig(model_id="test", embedding_dim=384)
        profile = create_hybrid_rank_profile(
            config,
            fusion_method="norm_linear",
        )

        # norm_linear should have global phase with normalize_linear
        assert profile.global_phase is not None
        assert "normalize_linear" in profile.global_phase.expression
        assert "bm25(text)" in profile.global_phase.expression
        assert "closeness" in profile.global_phase.expression

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

        # Test that old 'normalize' method is not valid
        with pytest.raises(ValueError, match="Unknown fusion_method"):
            create_hybrid_rank_profile(config, fusion_method="normalize")

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
        """Test that Vespa Cloud models are defined."""
        assert "nomic-ai-modernbert" in COMMON_MODELS
        assert "lightonai-modernbert-large" in COMMON_MODELS
        assert "alibaba-gte-modernbert" in COMMON_MODELS
        assert "e5-small-v2" in COMMON_MODELS
        assert "e5-base-v2" in COMMON_MODELS
        assert "e5-large-v2" in COMMON_MODELS
        assert "multilingual-e5-base" in COMMON_MODELS

    def test_e5_small_v2_config(self):
        """Test e5-small-v2 configuration."""
        config = COMMON_MODELS["e5-small-v2"]
        assert config.model_id == "e5-small-v2"
        assert config.embedding_dim == 384
        assert config.binarized is False
        assert config.max_tokens == 512
        assert config.query_prepend == "query: "
        assert config.document_prepend == "passage: "

    def test_nomic_ai_modernbert_config(self):
        """Test nomic-ai-modernbert configuration."""
        config = COMMON_MODELS["nomic-ai-modernbert"]
        assert config.model_id == "nomic-ai-modernbert"
        assert config.embedding_dim == 768
        assert config.binarized is False
        assert config.max_tokens == 8192
        assert config.transformer_output == "token_embeddings"
        assert config.query_prepend == "search_query: "
        assert config.document_prepend == "search_document: "

    def test_get_model_config_success(self):
        """Test getting a predefined model config."""
        config = get_model_config("e5-small-v2")
        assert config.model_id == "e5-small-v2"
        assert config.embedding_dim == 384

    def test_get_model_config_not_found(self):
        """Test that unknown model raises error."""
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_config("nonexistent-model")

        # Error message should mention list_models()
        try:
            get_model_config("nonexistent-model")
        except KeyError as e:
            assert "list_models()" in str(e)

    def test_get_model_config_fuzzy_matching(self):
        """Test that fuzzy matching provides suggestions for typos."""
        # Test with a typo that should match e5-small-v2
        with pytest.raises(KeyError, match="Did you mean"):
            get_model_config("e5-smal-v2")

        # Verify suggestion is included
        try:
            get_model_config("e5-smal-v2")
        except KeyError as e:
            error_msg = str(e)
            assert "e5-small-v2" in error_msg

        # Test with another typo
        with pytest.raises(KeyError, match="Did you mean"):
            get_model_config("nomic-modernbert")

        try:
            get_model_config("nomic-modernbert")
        except KeyError as e:
            error_msg = str(e)
            assert "nomic-ai-modernbert" in error_msg

    def test_list_models(self):
        """Test listing all available models."""
        models = list_models()

        # Should return a list
        assert isinstance(models, list)

        # Should contain all models from COMMON_MODELS
        assert len(models) == len(COMMON_MODELS)

        # Should be sorted
        assert models == sorted(models)

        # Should contain expected models
        assert "e5-small-v2" in models
        assert "nomic-ai-modernbert" in models
        assert "multilingual-e5-base" in models


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

    def test_complete_advanced_setup(self):
        """Test complete setup with URL-based model and explicit parameters."""
        config = ModelConfig(
            model_id="gte-multilingual",
            embedding_dim=768,
            model_url="https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/onnx/model_quantized.onnx",
            tokenizer_url="https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/tokenizer.json",
            transformer_output="token_embeddings",
            max_tokens=8192,
            query_prepend="Represent this sentence for searching relevant passages: ",
            document_prepend="passage: ",
        )

        component = create_embedder_component(config)
        field = create_embedding_field(config)
        semantic_profile = create_semantic_rank_profile(config)
        hybrid_profile = create_hybrid_rank_profile(config)

        # Verify component configuration
        assert component.id == "gte_multilingual"
        assert (
            len(component.parameters) == 5
        )  # transformer, tokenizer, max-tokens, transformer-output, prepend
        assert component.parameters[0].args["url"] == config.model_url
        assert component.parameters[1].args["url"] == config.tokenizer_url
        assert component.parameters[2].name == "max-tokens"
        assert component.parameters[2].children == "8192"
        assert component.parameters[3].name == "transformer-output"
        assert component.parameters[3].children == "token_embeddings"
        assert component.parameters[4].name == "prepend"
        assert len(component.parameters[4].children) == 2

        # Verify field configuration
        assert field.type == "tensor<float>(x[768])"
        assert field.ann.distance_metric == "angular"

        # Verify profiles
        assert semantic_profile.inputs[0][1] == "tensor<float>(x[768])"
        assert hybrid_profile.inputs[0][1] == "tensor<float>(x[768])"


class TestCommonModelsXMLGeneration:
    """Test that COMMON_MODELS generate XML matching Vespa Cloud documentation."""

    def test_nomic_ai_modernbert_xml(self):
        """Test nomic-ai-modernbert generates correct XML."""
        config = get_model_config("nomic-ai-modernbert")
        component = create_embedder_component(config)
        xml = component.to_xml_string(indent=1)

        expected = """<component id="nomic_ai_modernbert" type="hugging-face-embedder">
    <transformer-model model-id="nomic-ai-modernbert"/>
    <transformer-output>token_embeddings</transformer-output>
    <max-tokens>8192</max-tokens>
    <prepend>
        <query>search_query: </query>
        <document>search_document: </document>
    </prepend>
</component>"""

        assert compare_xml(
            xml, expected
        ), f"XML mismatch:\nGot:\n{xml}\n\nExpected:\n{expected}"

    def test_lightonai_modernbert_large_xml(self):
        """Test lightonai-modernbert-large generates correct XML."""
        config = get_model_config("lightonai-modernbert-large")
        component = create_embedder_component(config)
        xml = component.to_xml_string(indent=1)

        expected = """<component id="lightonai_modernbert_large" type="hugging-face-embedder">
    <transformer-model model-id="lightonai-modernbert-large"/>
    <max-tokens>8192</max-tokens>
    <prepend>
        <query>search_query:</query>
        <document>search_document:</document>
    </prepend>
</component>"""

        assert compare_xml(
            xml, expected
        ), f"XML mismatch:\nGot:\n{xml}\n\nExpected:\n{expected}"

    def test_alibaba_gte_modernbert_xml(self):
        """Test alibaba-gte-modernbert generates correct XML."""
        config = get_model_config("alibaba-gte-modernbert")
        component = create_embedder_component(config)
        xml = component.to_xml_string(indent=1)

        expected = """<component id="alibaba_gte_modernbert" type="hugging-face-embedder">
    <transformer-model model-id="alibaba-gte-modernbert"/>
    <max-tokens>8192</max-tokens>
    <pooling-strategy>cls</pooling-strategy>
</component>"""

        assert compare_xml(
            xml, expected
        ), f"XML mismatch:\nGot:\n{xml}\n\nExpected:\n{expected}"

    def test_e5_small_v2_xml(self):
        """Test e5-small-v2 generates correct XML."""
        config = get_model_config("e5-small-v2")
        component = create_embedder_component(config)
        xml = component.to_xml_string(indent=1)

        expected = """<component id="e5_small_v2" type="hugging-face-embedder">
    <transformer-model model-id="e5-small-v2"/>
    <max-tokens>512</max-tokens>
    <prepend>
        <query>query: </query>
        <document>passage: </document>
    </prepend>
</component>"""

        assert compare_xml(
            xml, expected
        ), f"XML mismatch:\nGot:\n{xml}\n\nExpected:\n{expected}"

    def test_e5_base_v2_xml(self):
        """Test e5-base-v2 generates correct XML."""
        config = get_model_config("e5-base-v2")
        component = create_embedder_component(config)
        xml = component.to_xml_string(indent=1)

        expected = """<component id="e5_base_v2" type="hugging-face-embedder">
    <transformer-model model-id="e5-base-v2"/>
    <max-tokens>512</max-tokens>
    <prepend>
        <query>query: </query>
        <document>passage: </document>
    </prepend>
</component>"""

        assert compare_xml(
            xml, expected
        ), f"XML mismatch:\nGot:\n{xml}\n\nExpected:\n{expected}"

    def test_e5_large_v2_xml(self):
        """Test e5-large-v2 generates correct XML."""
        config = get_model_config("e5-large-v2")
        component = create_embedder_component(config)
        xml = component.to_xml_string(indent=1)

        expected = """<component id="e5_large_v2" type="hugging-face-embedder">
    <transformer-model model-id="e5-large-v2"/>
    <max-tokens>512</max-tokens>
    <prepend>
        <query>query: </query>
        <document>passage: </document>
    </prepend>
</component>"""

        assert compare_xml(
            xml, expected
        ), f"XML mismatch:\nGot:\n{xml}\n\nExpected:\n{expected}"

    def test_multilingual_e5_base_xml(self):
        """Test multilingual-e5-base generates correct XML."""
        config = get_model_config("multilingual-e5-base")
        component = create_embedder_component(config)
        xml = component.to_xml_string(indent=1)

        expected = """<component id="multilingual_e5_base" type="hugging-face-embedder">
    <transformer-model model-id="multilingual-e5-base"/>
    <max-tokens>512</max-tokens>
    <prepend>
        <query>query: </query>
        <document>passage: </document>
    </prepend>
</component>"""

        assert compare_xml(
            xml, expected
        ), f"XML mismatch:\nGot:\n{xml}\n\nExpected:\n{expected}"


class TestCreateHybridPackage:
    """Test create_hybrid_package function."""

    def test_create_hybrid_package_multi_model_naming(self):
        """Test that multi-model setup creates properly namespaced fields and profiles."""
        package = create_hybrid_package(["e5-small-v2", "e5-base-v2"])

        # Check that we have 2 components
        assert len(package.components) == 2

        # Check that embedding fields are properly named
        embedding_fields = [
            f for f in package.schema.document.fields if "embedding" in f.name
        ]
        assert len(embedding_fields) == 2
        field_names = {f.name for f in embedding_fields}
        assert "embedding_e5_small_v2" in field_names
        assert "embedding_e5_base_v2" in field_names

        # Check that rank profiles are properly namespaced
        profile_names = set(package.schema.rank_profiles.keys())

        # Should have match-only profile
        assert "match-only" in profile_names

        # Should have profiles for each model with proper suffixes
        assert "bm25_e5_small_v2" in profile_names
        assert "bm25_e5_base_v2" in profile_names
        assert "semantic_e5_small_v2" in profile_names
        assert "semantic_e5_base_v2" in profile_names
        assert "fusion_e5_small_v2" in profile_names
        assert "fusion_e5_base_v2" in profile_names
        assert "atan_norm_e5_small_v2" in profile_names
        assert "atan_norm_e5_base_v2" in profile_names
        assert "norm_linear_e5_small_v2" in profile_names
        assert "norm_linear_e5_base_v2" in profile_names
