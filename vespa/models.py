from __future__ import annotations
import re
from difflib import get_close_matches
from typing import Dict, List, Optional, Union, Literal
from dataclasses import dataclass
from vespa.package import (
    ApplicationPackage,
    Component,
    Parameter,
    Field,
    HNSW,
    RankProfile,
    Function,
    Schema,
    Document,
    FieldSet,
)

PoolingStrategy = Literal["mean", "cls", "none"]
DistanceMetric = Literal[
    "angular", "euclidean", "dotproduct", "hamming", "prenormalized-angular"
]
FusionMethod = Literal["rrf", "atan_norm", "norm_linear"]


# ============================================================================
# Model Configuration Utilities
# ============================================================================
# The following section provides utilities to easily configure and run
# different evaluations for different embedding models, handling differences
# in model dimensions, tokenizers, and binary vs. float embeddings.
# ============================================================================


def sanitize_component_id(model_id: str) -> str:
    """
    Sanitize a model ID to create a valid Vespa component identifier.

    Vespa component IDs must match the pattern [a-zA-Z][a-zA-Z0-9_]*
    (start with a letter, followed by letters, digits, or underscores).

    Args:
        model_id: The model identifier to sanitize

    Returns:
        A valid Vespa component ID

    Example:
        >>> sanitize_component_id("e5-small-v2")
        'e5_small_v2'
        >>> sanitize_component_id("sentence-transformers/all-MiniLM-L6-v2")
        'sentence_transformers_all_MiniLM_L6_v2'
        >>> sanitize_component_id("model.v1.0")
        'model_v1_0'
        >>> sanitize_component_id("123-model")
        'model_123_model'
    """
    # Replace any non-alphanumeric characters (except underscores) with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", model_id)

    # Ensure it starts with a letter (if it starts with digit/underscore, prepend 'model_')
    if not sanitized[0].isalpha():
        sanitized = "model_" + sanitized

    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    return sanitized


@dataclass
class ModelConfig:
    """
    Configuration for an embedding model.

    This class encapsulates all model-specific parameters that affect
    the Vespa schema, component configuration, and ranking expressions.

    Attributes:
        model_id: The model identifier (e.g., 'e5-small-v2', 'snowflake-arctic-embed-xs')
        embedding_dim: The dimension of the embedding vectors (e.g., 384, 768)
        tokenizer_id: The tokenizer model identifier (if different from model_id)
        binarized: Whether the embeddings should be binarized (packed to bits)
        component_id: The ID to use for the Vespa component (defaults to sanitized model_id)
        model_path: Optional local path to the model file
        tokenizer_path: Optional local path to the tokenizer file
        model_url: Optional URL to the ONNX model file (alternative to model_id)
        tokenizer_url: Optional URL to the tokenizer file (alternative to tokenizer_id)
        max_tokens: Maximum number of tokens accepted by the transformer model (default: 512)
        transformer_input_ids: Name/identifier for transformer input IDs (default: "input_ids")
        transformer_attention_mask: Name/identifier for transformer attention mask (default: "attention_mask")
        transformer_token_type_ids: Name/identifier for transformer token type IDs (default: "token_type_ids")
            Set to None to disable token_type_ids
        transformer_output: Name/identifier for transformer output (default: "last_hidden_state")
        pooling_strategy: How to pool output vectors ("mean", "cls", or "none") (default: "mean")
        normalize: Whether to normalize output to unit length (default: False)
        query_prepend: Optional instruction to prepend to query text
        document_prepend: Optional instruction to prepend to document text
    """

    model_id: str
    embedding_dim: int
    tokenizer_id: Optional[str] = None
    binarized: bool = False
    component_id: Optional[str] = None
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    model_url: Optional[str] = None
    tokenizer_url: Optional[str] = None
    max_tokens: Optional[int] = None
    transformer_input_ids: Optional[str] = None
    transformer_attention_mask: Optional[str] = None
    transformer_token_type_ids: Optional[str] = None
    transformer_output: Optional[str] = None
    pooling_strategy: Optional[PoolingStrategy] = None
    normalize: Optional[bool] = None
    query_prepend: Optional[str] = None
    document_prepend: Optional[str] = None

    def __post_init__(self):
        """Set defaults and validate configuration."""
        if self.tokenizer_id is None:
            # Use the same ID for tokenizer if not specified
            self.tokenizer_id = self.model_id

        if self.component_id is None:
            # Create a valid component ID from model_id
            self.component_id = sanitize_component_id(self.model_id)

        # Validate embedding dimension
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )

        # Validate binarized embeddings require dimension divisible by 8
        if self.binarized and self.embedding_dim % 8 != 0:
            raise ValueError(
                f"binarized embeddings require embedding_dim divisible by 8, "
                f"got {self.embedding_dim} which would be truncated to {self.embedding_dim // 8} bytes"
            )

        # Validate pooling strategy (type checker handles valid values, but runtime check for safety)
        if self.pooling_strategy is not None:
            valid_strategies = ["mean", "cls", "none"]
            if self.pooling_strategy not in valid_strategies:
                raise ValueError(
                    f"pooling_strategy must be one of {valid_strategies}, got {self.pooling_strategy}"
                )


def create_embedder_component(config: ModelConfig) -> Component:
    """
    Create a Vespa hugging-face-embedder component from a model configuration.

    Args:
        config: ModelConfig instance with model parameters

    Returns:
        Component: A Vespa Component configured as a hugging-face-embedder

    Example:
        >>> config = ModelConfig(model_id="e5-small-v2", embedding_dim=384)
        >>> component = create_embedder_component(config)
        >>> component.id
        'e5_small_v2'

        >>> # Example with URL-based model and custom parameters
        >>> config = ModelConfig(
        ...     model_id="gte-multilingual",
        ...     embedding_dim=768,
        ...     model_url="https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/onnx/model_quantized.onnx",
        ...     tokenizer_url="https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/tokenizer.json",
        ...     transformer_output="token_embeddings",
        ...     max_tokens=8192,
        ...     query_prepend="Represent this sentence for searching relevant passages: ",
        ...     document_prepend="passage: ",
        ... )
        >>> component = create_embedder_component(config)
        >>> component.id
        'gte_multilingual'
    """
    parameters = []

    # Add transformer model parameter
    if config.model_url:
        transformer_config = {"url": config.model_url}
    elif config.model_path:
        transformer_config = {"path": config.model_path}
    else:
        transformer_config = {"model-id": config.model_id}
    parameters.append(Parameter("transformer-model", transformer_config))

    # Add tokenizer model parameter
    if config.tokenizer_url:
        tokenizer_config = {"url": config.tokenizer_url}
    elif config.tokenizer_path:
        tokenizer_config = {"path": config.tokenizer_path}
    else:
        tokenizer_config = None
    if tokenizer_config is not None:
        parameters.append(Parameter("tokenizer-model", tokenizer_config))
    # Add optional huggingface embedder parameters
    if config.max_tokens is not None:
        parameters.append(
            Parameter("max-tokens", args={}, children=str(config.max_tokens))
        )

    if config.transformer_input_ids is not None:
        parameters.append(
            Parameter(
                "transformer-input-ids", args={}, children=config.transformer_input_ids
            )
        )

    if config.transformer_attention_mask is not None:
        parameters.append(
            Parameter(
                "transformer-attention-mask",
                args={},
                children=config.transformer_attention_mask,
            )
        )

    if config.transformer_token_type_ids is not None:
        # Empty element to disable token_type_ids
        if config.transformer_token_type_ids == "":
            parameters.append(
                Parameter("transformer-token-type-ids", args={}, children=None)
            )
        else:
            parameters.append(
                Parameter(
                    "transformer-token-type-ids",
                    args={},
                    children=config.transformer_token_type_ids,
                )
            )

    if config.transformer_output is not None:
        parameters.append(
            Parameter("transformer-output", args={}, children=config.transformer_output)
        )

    if config.pooling_strategy is not None:
        parameters.append(
            Parameter("pooling-strategy", args={}, children=config.pooling_strategy)
        )

    if config.normalize is not None:
        parameters.append(
            Parameter("normalize", args={}, children=str(config.normalize).lower())
        )

    # Add prepend instructions if specified
    if config.query_prepend is not None or config.document_prepend is not None:
        prepend_children = []
        if config.query_prepend is not None:
            prepend_children.append(
                Parameter("query", args={}, children=config.query_prepend)
            )
        if config.document_prepend is not None:
            prepend_children.append(
                Parameter("document", args={}, children=config.document_prepend)
            )
        parameters.append(Parameter("prepend", args={}, children=prepend_children))

    return Component(
        id=config.component_id,
        type="hugging-face-embedder",
        parameters=parameters,
    )


def create_embedding_field(
    config: ModelConfig,
    field_name: str = "embedding",
    indexing: Optional[List[str]] = None,
    distance_metric: Optional[DistanceMetric] = None,
    embedder_id: Optional[str] = None,
) -> Field:
    """
    Create a Vespa embedding field from a model configuration.

    The field type and indexing statement are automatically configured based on
    whether the embeddings are binarized.

    Args:
        config: ModelConfig instance with model parameters
        field_name: Name of the embedding field (default: "embedding")
        indexing: Custom indexing statement (default: auto-generated based on config)
        distance_metric: Distance metric for HNSW (default: "hamming" for binarized, "angular" for float)
        embedder_id: Embedder ID to use in the indexing statement (default: uses config.component_id)

    Returns:
        Field: A Vespa Field configured for embeddings

    Example:
        >>> config = ModelConfig(model_id="e5-small-v2", embedding_dim=384, binarized=False)
        >>> field = create_embedding_field(config)
        >>> field.type
        'tensor<float>(x[384])'

        >>> config_binary = ModelConfig(model_id="bge-m3", embedding_dim=1024, binarized=True)
        >>> field_binary = create_embedding_field(config_binary)
        >>> field_binary.type
        'tensor<int8>(x[128])'
    """
    # Determine embedder ID to use
    embedder_id = embedder_id or config.component_id

    # Determine field type based on binarization
    if config.binarized:
        # For binarized embeddings, we pack 8 bits into each int8
        packed_dim = config.embedding_dim // 8
        field_type = f"tensor<int8>(x[{packed_dim}])"
        default_distance_metric = "hamming"

        # Default indexing for binarized: pack bits and index
        if indexing is None:
            indexing = [
                "input text",
                f"embed {embedder_id}",
                "pack_bits",
                "index",
                "attribute",
            ]
    else:
        # Regular float embeddings
        field_type = f"tensor<float>(x[{config.embedding_dim}])"
        default_distance_metric = "angular"

        # Default indexing for float embeddings
        if indexing is None:
            indexing = [
                "input text",
                f"embed {embedder_id}",
                "index",
                "attribute",
            ]

    # Use provided distance metric or default
    distance_metric = distance_metric or default_distance_metric

    return Field(
        name=field_name,
        type=field_type,
        indexing=indexing,
        ann=HNSW(distance_metric=distance_metric),
        is_document_field=False,
    )


def create_semantic_rank_profile(
    config: ModelConfig,
    profile_name: str = "semantic",
    embedding_field: str = "embedding",
    query_tensor: str = "q",
) -> RankProfile:
    """
    Create a semantic ranking profile based on model configuration.

    The ranking expression is automatically configured to use hamming distance
    for binarized embeddings or cosine similarity for float embeddings.

    Args:
        config: ModelConfig instance with model parameters
        profile_name: Name of the rank profile (default: "semantic")
        embedding_field: Name of the embedding field (default: "embedding")
        query_tensor: Name of the query tensor (default: "q")

    Returns:
        RankProfile: A Vespa RankProfile configured for semantic search

    Example:
        >>> config = ModelConfig(model_id="e5-small-v2", embedding_dim=384, binarized=False)
        >>> profile = create_semantic_rank_profile(config)
        >>> profile.name
        'semantic'
    """
    # Determine tensor type for query input
    if config.binarized:
        packed_dim = config.embedding_dim // 8
        tensor_type = f"tensor<int8>(x[{packed_dim}])"

        # For binarized, use hamming distance
        # Note: closeness() with hamming distance returns similarity (lower is more similar)
        # We use negation or subtraction to convert to a score where higher is better
        similarity_expr = f"1/(1 + closeness(field, {embedding_field}))"
    else:
        tensor_type = f"tensor<float>(x[{config.embedding_dim}])"

        # For float embeddings, use angular distance (cosine similarity)
        similarity_expr = f"closeness(field, {embedding_field})"

    return RankProfile(
        name=profile_name,
        inputs=[(f"query({query_tensor})", tensor_type)],
        functions=[Function(name="similarity", expression=similarity_expr)],
        first_phase="similarity",
        match_features=["similarity"],
    )


def create_hybrid_rank_profile(
    config: ModelConfig,
    profile_name: str = "fusion",
    base_profile: str = "bm25",
    embedding_field: str = "embedding",
    query_tensor: str = "q",
    fusion_method: FusionMethod = "rrf",
    global_rerank_count: int = 1000,
    second_phase_rerank_count: Optional[int] = None,
    first_phase_rerank_count: Optional[int] = None,
) -> RankProfile:
    """
    Create a hybrid ranking profile combining BM25 and semantic search.

    Args:
        config: ModelConfig instance with model parameters
        profile_name: Name of the rank profile (default: "fusion")
        base_profile: Name of the BM25 profile to inherit from (default: "bm25")
        embedding_field: Name of the embedding field (default: "embedding")
        query_tensor: Name of the query tensor (default: "q")
        fusion_method: Fusion method - "rrf" for reciprocal rank fusion, "atan_norm" for atan-normalized sum in first phase, or "norm_linear" for linear normalization in global phase.
        global_rerank_count: Number of hits to rerank in global phase (default: 1000)
        second_phase_rerank_count: Number of hits to rerank in second phase (default: None, only used with fusion methods that have second phase)
        first_phase_rerank_count: Number of hits to rerank in first phase (default: None)

    Returns:
        RankProfile: A Vespa RankProfile configured for hybrid search

    Example:
        >>> config = ModelConfig(model_id="e5-small-v2", embedding_dim=384)
        >>> profile = create_hybrid_rank_profile(config)
        >>> profile.name
        'fusion'
    """
    # Import necessary classes here to avoid circular dependency
    from vespa.package import (
        GlobalPhaseRanking,
        SecondPhaseRanking,
        FirstPhaseRanking,
    )

    # Determine tensor type for query input
    if config.binarized:
        packed_dim = config.embedding_dim // 8
        tensor_type = f"tensor<int8>(x[{packed_dim}])"
        similarity_expr = f"1/(1 + closeness(field, {embedding_field}))"
    else:
        tensor_type = f"tensor<float>(x[{config.embedding_dim}])"
        similarity_expr = f"closeness(field, {embedding_field})"

    # Choose global phase expression based on fusion method
    if fusion_method == "rrf":
        global_expr = (
            f"reciprocal_rank_fusion(bm25text, closeness(field, {embedding_field}))"
        )
        functions = [Function(name="similarity", expression=similarity_expr)]
        first_phase_expr = "similarity"
        match_features_list = ["similarity", "bm25text"]
        global_phase = GlobalPhaseRanking(
            expression=global_expr,
            rerank_count=global_rerank_count,
        )
    elif fusion_method == "atan_norm":
        # Use atan normalization in first phase only (no global phase)
        functions = [
            Function(
                name="scale",
                args=["val"],
                expression="2*atan(val)/(3.14159)",
            ),
            Function(name="normalized_bm25", expression="scale(bm25(text))"),
            Function(name="cos_sim", expression=f"closeness(field, {embedding_field})"),
        ]
        first_phase_expr = "normalized_bm25 + cos_sim"
        match_features_list = ["cos_sim", "normalized_bm25"]
        global_phase = None
    elif fusion_method == "norm_linear":
        # Use linear normalization in global phase (no atan)
        global_expr = f"normalize_linear(bm25(text)) + normalize_linear(closeness(field, {embedding_field}))"
        functions = [
            Function(name="cos_sim", expression=f"closeness(field, {embedding_field})"),
        ]
        first_phase_expr = "cos_sim"
        match_features_list = ["cos_sim", "bm25(text)"]
        global_phase = GlobalPhaseRanking(
            expression=global_expr,
            rerank_count=global_rerank_count,
        )
    else:
        raise ValueError(
            f"Unknown fusion_method: {fusion_method}. Use 'rrf', 'atan_norm', or 'norm_linear'."
        )

    first_phase_obj = FirstPhaseRanking(
        expression=first_phase_expr, keep_rank_count=first_phase_rerank_count
    )

    second_phase = None
    if second_phase_rerank_count is not None:
        # For second phase, we can rerank with a more expensive expression
        # For now, use the same expression as first phase
        second_phase = SecondPhaseRanking(
            expression=first_phase_expr, rerank_count=second_phase_rerank_count
        )

    return RankProfile(
        name=profile_name,
        inherits=base_profile,
        inputs=[(f"query({query_tensor})", tensor_type)],
        functions=functions,
        first_phase=first_phase_obj,
        second_phase=second_phase,
        global_phase=global_phase,
        match_features=match_features_list,
    )


# Predefined model configurations for Vespa Cloud models
# Based on https://cloud.vespa.ai/en/model-hub
COMMON_MODELS: Dict[str, ModelConfig] = {
    # Huggingface Embedder models
    "nomic-ai-modernbert": ModelConfig(
        model_id="nomic-ai-modernbert",
        embedding_dim=768,
        binarized=False,
        transformer_output="token_embeddings",
        max_tokens=8192,
        query_prepend="search_query: ",
        document_prepend="search_document: ",
    ),
    "lightonai-modernbert-large": ModelConfig(
        model_id="lightonai-modernbert-large",
        embedding_dim=1024,
        binarized=False,
        max_tokens=8192,
        query_prepend="search_query: ",
        document_prepend="search_document: ",
    ),
    "alibaba-gte-modernbert": ModelConfig(
        model_id="alibaba-gte-modernbert",
        embedding_dim=768,
        binarized=False,
        max_tokens=8192,
        pooling_strategy="cls",
    ),
    "e5-small-v2": ModelConfig(
        model_id="e5-small-v2",
        embedding_dim=384,
        binarized=False,
        max_tokens=512,
        query_prepend="query: ",
        document_prepend="passage: ",
    ),
    "e5-base-v2": ModelConfig(
        model_id="e5-base-v2",
        embedding_dim=768,
        binarized=False,
        max_tokens=512,
        query_prepend="query: ",
        document_prepend="passage: ",
    ),
    "e5-large-v2": ModelConfig(
        model_id="e5-large-v2",
        embedding_dim=1024,
        binarized=False,
        max_tokens=512,
        query_prepend="query: ",
        document_prepend="passage: ",
    ),
    "multilingual-e5-base": ModelConfig(
        model_id="multilingual-e5-base",
        embedding_dim=768,
        binarized=False,
        max_tokens=512,
        query_prepend="query: ",
        document_prepend="passage: ",
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get a predefined model configuration by name.

    Args:
        model_name: Name of a predefined model

    Returns:
        ModelConfig: The model configuration

    Raises:
        KeyError: If the model name is not found

    Example:
        >>> config = get_model_config("e5-small-v2")
        >>> config.embedding_dim
        384
    """
    if model_name not in COMMON_MODELS:
        # Provide fuzzy matching suggestions for better UX
        suggestions = get_close_matches(
            model_name, COMMON_MODELS.keys(), n=3, cutoff=0.6
        )

        if suggestions:
            suggestions_str = ", ".join(f"'{s}'" for s in suggestions)
            raise KeyError(
                f"Unknown model '{model_name}'. Did you mean: {suggestions_str}? "
                f"Use list_models() to see all available models."
            )
        else:
            raise KeyError(
                f"Unknown model '{model_name}'. "
                f"Use list_models() to see all {len(COMMON_MODELS)} available models."
            )
    return COMMON_MODELS[model_name]


def list_models() -> List[str]:
    """
    List all available predefined model configurations.

    Returns:
        List of model names that can be used with get_model_config()

    Example:
        >>> models = list_models()
        >>> 'e5-small-v2' in models
        True
        >>> 'nomic-ai-modernbert' in models
        True
    """
    return sorted(COMMON_MODELS.keys())


def _create_model_profiles(
    config: ModelConfig,
    embedding_field_name: str,
    profile_suffix: str,
    query_tensor: str,
) -> List[RankProfile]:
    """
    Helper function to create all rank profiles for a single model.

    Args:
        config: ModelConfig instance
        embedding_field_name: Name of the embedding field
        profile_suffix: Suffix to add to profile names (empty string for single model)
        query_tensor: Name of the query tensor

    Returns:
        List of RankProfile objects for this model
    """
    profiles = []

    # Base BM25 rank profile
    embedding_type = (
        f"tensor<int8>(x[{config.embedding_dim // 8}])"
        if config.binarized
        else f"tensor<float>(x[{config.embedding_dim}])"
    )
    bm25_profile = RankProfile(
        name=f"bm25{profile_suffix}",
        inputs=[(f"query({query_tensor})", embedding_type)],
        functions=[Function(name="bm25text", expression="bm25(text)")],
        first_phase="bm25text",
        match_features=["bm25text"],
    )
    profiles.append(bm25_profile)

    # Semantic search rank profile
    semantic_profile = create_semantic_rank_profile(
        config,
        profile_name=f"semantic{profile_suffix}",
        embedding_field=embedding_field_name,
        query_tensor=query_tensor,
    )
    profiles.append(semantic_profile)

    # Hybrid rank profiles with different fusion methods
    fusion_methods: List[FusionMethod] = ["rrf", "atan_norm", "norm_linear"]
    for method in fusion_methods:
        profile_name = (
            f"fusion{profile_suffix}"
            if method == "rrf"
            else f"{method}{profile_suffix}"
        )
        hybrid_profile = create_hybrid_rank_profile(
            config,
            profile_name=profile_name,
            base_profile=f"bm25{profile_suffix}",
            embedding_field=embedding_field_name,
            query_tensor=query_tensor,
            fusion_method=method,
        )
        profiles.append(hybrid_profile)

    return profiles


def create_hybrid_package(
    models: Union[str, ModelConfig, List[Union[str, ModelConfig]]],
    app_name: str = "hybridapp",
    schema_name: str = "doc",
) -> ApplicationPackage:
    """
    Create a Vespa application package configured for hybrid search evaluation.

    This function creates a complete Vespa application package with all necessary
    components, fields, and rank profiles for evaluation. It supports single or
    multiple embedding models, automatically handling naming conflicts by using
    model-specific field and component names.

    Args:
        models: Single model or list of models to configure. Each can be:
            - A string model name (e.g., "e5-small-v2") to use a predefined config
            - A ModelConfig instance for custom configuration
        app_name: Name of the application (default: "hybridapp")
        schema_name: Name of the schema (default: "doc")

    Returns:
        ApplicationPackage: Configured Vespa application package with:
            - Components for each embedding model
            - Embedding fields for each model (named "embedding" for single model,
              "embedding_{component_id}" for multiple models)
            - BM25 and semantic rank profiles for each model
            - Hybrid rank profiles (RRF, atan_norm, norm_linear) for each model
            - A match-only rank profile for baseline evaluation

    Raises:
        ValueError: If models list is empty
        KeyError: If a model name is not found in COMMON_MODELS

    Example:
        >>> # Single model by name
        >>> package = create_hybrid_package("e5-small-v2")
        >>> len(package.components)
        1
        >>> package.schema.document.fields[2].name
        'embedding'

        >>> # Single model with custom config
        >>> config = ModelConfig(model_id="my-model", embedding_dim=512)
        >>> package = create_hybrid_package(config)
        >>> package.schema.document.fields[2].name
        'embedding'

        >>> # Multiple models - creates separate fields and profiles for each
        >>> package = create_hybrid_package(["e5-small-v2", "e5-base-v2"])
        >>> len(package.components)
        2
        >>> # Fields will be named: embedding_e5_small_v2, embedding_e5_base_v2
        >>> field_names = [f.name for f in package.schema.document.fields if f.name.startswith('embedding')]
        >>> len(field_names)
        2

        >>> # Multiple models with mixed configs
        >>> custom = ModelConfig(model_id="custom-model", embedding_dim=384)
        >>> package = create_hybrid_package(["e5-small-v2", custom])
        >>> len(package.components)
        2
    """
    # Normalize input to a list of ModelConfig objects
    if isinstance(models, (str, ModelConfig)):
        model_configs = [models]
    else:
        model_configs = list(models)

    if not model_configs:
        raise ValueError("At least one model must be provided")

    # Convert string model names to ModelConfig objects
    resolved_configs = []
    for model in model_configs:
        if isinstance(model, str):
            resolved_configs.append(get_model_config(model))
        else:
            resolved_configs.append(model)

    # Determine if we have multiple models (affects naming)
    is_multi_model = len(resolved_configs) > 1

    # Build components, fields, and profiles for each model
    all_components = []
    all_embedding_fields = []
    all_rank_profiles = []
    match_only_inputs = []

    for config in resolved_configs:
        # Create unique identifiers for multi-model setup
        profile_suffix = f"_{config.component_id}" if is_multi_model else ""
        embedding_field_name = (
            f"embedding_{config.component_id}" if is_multi_model else "embedding"
        )
        query_tensor = f"q{profile_suffix}"

        # Create and collect component
        all_components.append(create_embedder_component(config))

        # Create and collect embedding field
        embedding_field = create_embedding_field(
            config, field_name=embedding_field_name, embedder_id=config.component_id
        )
        all_embedding_fields.append(embedding_field)

        # Track query inputs for match-only profile
        match_only_inputs.append((f"query({query_tensor})", embedding_field.type))

        # Create and collect all rank profiles for this model
        all_rank_profiles.extend(
            _create_model_profiles(
                config, embedding_field_name, profile_suffix, query_tensor
            )
        )

    # Create match-only profile with inputs for all models
    match_only_profile = RankProfile(
        name="match-only",
        inputs=match_only_inputs,
        first_phase="random",
    )

    # Build the schema with all fields
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
            ]
            + all_embedding_fields
        ),
        fieldsets=[FieldSet(name="default", fields=["text"])],
        rank_profiles=[match_only_profile] + all_rank_profiles,
    )

    # Create the application package
    return ApplicationPackage(
        name=app_name,
        schema=[schema],
        components=all_components,
    )
