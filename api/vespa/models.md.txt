## `vespa.models`

### `ModelConfig(model_id, embedding_dim, tokenizer_id=None, binarized=False, embedding_field_type='float', distance_metric=None, component_id=None, model_path=None, tokenizer_path=None, model_url=None, tokenizer_url=None, max_tokens=None, transformer_input_ids=None, transformer_attention_mask=None, transformer_token_type_ids=None, transformer_output=None, pooling_strategy=None, normalize=None, query_prepend=None, document_prepend=None, validate_urls=False)`

Configuration for an embedding model.

This class encapsulates all model-specific parameters that affect the Vespa schema, component configuration, and ranking expressions.

Attributes:

| Name                         | Type                        | Description                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_id`                   | `str`                       | The model identifier (e.g., 'e5-small-v2', 'snowflake-arctic-embed-xs')                                                                                                                                                                                                                                                                                                                        |
| `embedding_dim`              | `int`                       | The dimension of the embedding vectors (e.g., 384, 768). When binarized=True, specify the original model dimension - it will be automatically divided by 8 for storage (e.g., 1024 -> 128 bytes).                                                                                                                                                                                              |
| `tokenizer_id`               | `Optional[str]`             | The tokenizer model identifier (if different from model_id)                                                                                                                                                                                                                                                                                                                                    |
| `binarized`                  | `bool`                      | Whether the embeddings should be binarized (packed to bits). When True, overrides embedding_field_type to int8 and embedding_dim must be divisible by 8.                                                                                                                                                                                                                                       |
| `embedding_field_type`       | `EmbeddingFieldType`        | Tensor cell type for embeddings (default: "float"). Note: When binarized=True, this is automatically overridden to "int8". Options: - "double": 64-bit float (highest precision, highest memory) - "float": 32-bit float (good balance) - "bfloat16": 16-bit brain float (reduced memory, good for large scale) - "int8": 8-bit integer (quantized, or used automatically when binarized=True) |
| `distance_metric`            | `Optional[DistanceMetric]`  | Distance metric for HNSW index (default: None, auto-set based on binarized). When binarized=True, automatically set to "hamming". When binarized=False and not specified, defaults to "angular". Options: - "angular": Cosine similarity - "hamming": Hamming distance (required for binarized embeddings) - "euclidean", "dotproduct", "prenormalized-angular", "geodegrees"                  |
| `component_id`               | `Optional[str]`             | The ID to use for the Vespa component (defaults to sanitized model_id)                                                                                                                                                                                                                                                                                                                         |
| `model_path`                 | `Optional[str]`             | Optional local path to the model file                                                                                                                                                                                                                                                                                                                                                          |
| `tokenizer_path`             | `Optional[str]`             | Optional local path to the tokenizer file                                                                                                                                                                                                                                                                                                                                                      |
| `model_url`                  | `Optional[str]`             | Optional URL to the ONNX model file (alternative to model_id)                                                                                                                                                                                                                                                                                                                                  |
| `tokenizer_url`              | `Optional[str]`             | Optional URL to the tokenizer file (alternative to tokenizer_id)                                                                                                                                                                                                                                                                                                                               |
| `max_tokens`                 | `Optional[int]`             | Maximum number of tokens accepted by the transformer model. Optional, if not set the Vespa embedder uses its internal default (512).                                                                                                                                                                                                                                                           |
| `transformer_input_ids`      | `Optional[str]`             | Name/identifier for transformer input IDs. Optional, if not set the Vespa embedder uses its internal default ("input_ids").                                                                                                                                                                                                                                                                    |
| `transformer_attention_mask` | `Optional[str]`             | Name/identifier for transformer attention mask. Optional, if not set the Vespa embedder uses its internal default ("attention_mask").                                                                                                                                                                                                                                                          |
| `transformer_token_type_ids` | `Optional[str]`             | Name/identifier for transformer token type IDs. Optional, if not set the Vespa embedder uses its internal default ("token_type_ids"). Set to empty string "" to explicitly disable token_type_ids.                                                                                                                                                                                             |
| `transformer_output`         | `Optional[str]`             | Name/identifier for transformer output. Optional, if not set the Vespa embedder uses its internal default ("last_hidden_state").                                                                                                                                                                                                                                                               |
| `pooling_strategy`           | `Optional[PoolingStrategy]` | How to pool output vectors ("mean", "cls", or "none"). Optional, if not set the Vespa embedder uses its internal default ("mean").                                                                                                                                                                                                                                                             |
| `normalize`                  | `Optional[bool]`            | Whether to normalize output to unit length. Optional, if not set the Vespa embedder uses its internal default (False).                                                                                                                                                                                                                                                                         |
| `query_prepend`              | `Optional[str]`             | Optional instruction to prepend to query text                                                                                                                                                                                                                                                                                                                                                  |
| `document_prepend`           | `Optional[str]`             | Optional instruction to prepend to document text                                                                                                                                                                                                                                                                                                                                               |
| `validate_urls`              | `bool`                      | Whether to validate URLs by checking they return HTTP 200 (default: False)                                                                                                                                                                                                                                                                                                                     |

#### `__post_init__()`

Set defaults and validate configuration.

#### `to_dict(include_none=False)`

Convert the ModelConfig to a dictionary for serialization.

Parameters:

| Name           | Type   | Description                                              | Default |
| -------------- | ------ | -------------------------------------------------------- | ------- |
| `include_none` | `bool` | If True, include fields with None values. Default False. | `False` |

Returns:

| Type             | Description                                   |
| ---------------- | --------------------------------------------- |
| `Dict[str, Any]` | Dict with all model configuration attributes. |

Example

> > > config = ModelConfig(model_id="e5-small-v2", embedding_dim=384) d = config.to_dict() d["model_id"] 'e5-small-v2' d["embedding_dim"] 384 d["binarized"] False

### `ApplicationPackageWithQueryFunctions(query_functions=None, **kwargs)`

Bases: `ApplicationPackage`

#### `get_query_functions()`

Get the query functions for this application package.

Returns:

| Type                                    | Description             |
| --------------------------------------- | ----------------------- |
| `Dict[str, Callable[[str, int], dict]]` | Dict of query functions |

### `sanitize_component_id(model_id)`

Sanitize a model ID to create a valid Vespa component identifier.

Vespa component IDs must match the pattern a-zA-Z\* (start with a letter, followed by letters, digits, or underscores).

Parameters:

| Name       | Type  | Description                      | Default    |
| ---------- | ----- | -------------------------------- | ---------- |
| `model_id` | `str` | The model identifier to sanitize | *required* |

Returns:

| Type  | Description                |
| ----- | -------------------------- |
| `str` | A valid Vespa component ID |

Example

> > > sanitize_component_id("e5-small-v2") 'e5_small_v2' sanitize_component_id("sentence-transformers/all-MiniLM-L6-v2") 'sentence_transformers_all_MiniLM_L6_v2' sanitize_component_id("model.v1.0") 'model_v1_0' sanitize_component_id("123-model") 'model_123_model'

### `create_embedder_component(config)`

Create a Vespa hugging-face-embedder component from a model configuration.

Parameters:

| Name     | Type          | Description                                | Default    |
| -------- | ------------- | ------------------------------------------ | ---------- |
| `config` | `ModelConfig` | ModelConfig instance with model parameters | *required* |

Returns:

| Name        | Type        | Description                                             |
| ----------- | ----------- | ------------------------------------------------------- |
| `Component` | `Component` | A Vespa Component configured as a hugging-face-embedder |

Example

> > > config = ModelConfig(model_id="e5-small-v2", embedding_dim=384) component = create_embedder_component(config) component.id 'e5_small_v2'
> > >
> > > #### Example with URL-based model and custom parameters
> > >
> > > config = ModelConfig( ... model_id="gte-multilingual", ... embedding_dim=768, ... model_url="https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/onnx/model_quantized.onnx", ... tokenizer_url="https://huggingface.co/onnx-community/gte-multilingual-base/resolve/main/tokenizer.json", ... transformer_output="token_embeddings", ... max_tokens=8192, ... query_prepend="Represent this sentence for searching relevant passages: ", ... document_prepend="passage: ", ... ) component = create_embedder_component(config) component.id 'gte_multilingual'

### `create_embedding_field(config, field_name='embedding', indexing=None, distance_metric=None, embedder_id=None)`

Create a Vespa embedding field from a model configuration.

The field type and indexing statement are automatically configured based on whether the embeddings are binarized.

Parameters:

| Name              | Type                       | Description                                                                      | Default       |
| ----------------- | -------------------------- | -------------------------------------------------------------------------------- | ------------- |
| `config`          | `ModelConfig`              | ModelConfig instance with model parameters                                       | *required*    |
| `field_name`      | `str`                      | Name of the embedding field (default: "embedding")                               | `'embedding'` |
| `indexing`        | `Optional[List[str]]`      | Custom indexing statement (default: auto-generated based on config)              | `None`        |
| `distance_metric` | `Optional[DistanceMetric]` | Distance metric for HNSW (default: "hamming" for binarized, "angular" for float) | `None`        |
| `embedder_id`     | `Optional[str]`            | Embedder ID to use in the indexing statement (default: uses config.component_id) | `None`        |

Returns:

| Name    | Type    | Description                             |
| ------- | ------- | --------------------------------------- |
| `Field` | `Field` | A Vespa Field configured for embeddings |

Example

> > > config = ModelConfig(model_id="e5-small-v2", embedding_dim=384) field = create_embedding_field(config) field.type 'tensor(x[384])'
> > >
> > > config_float = ModelConfig(model_id="e5-small-v2", embedding_dim=384, embedding_field_type="float") field_float = create_embedding_field(config_float) field_float.type 'tensor(x[384])'
> > >
> > > config_binary = ModelConfig(model_id="bge-m3", embedding_dim=1024, binarized=True) field_binary = create_embedding_field(config_binary) field_binary.type 'tensor(x[128])'

### `create_semantic_rank_profile(config, profile_name='semantic', embedding_field='embedding', query_tensor='q')`

Create a semantic ranking profile based on model configuration.

The ranking expression is automatically configured to use hamming distance for binarized embeddings or cosine similarity for float embeddings.

Parameters:

| Name              | Type          | Description                                        | Default       |
| ----------------- | ------------- | -------------------------------------------------- | ------------- |
| `config`          | `ModelConfig` | ModelConfig instance with model parameters         | *required*    |
| `profile_name`    | `str`         | Name of the rank profile (default: "semantic")     | `'semantic'`  |
| `embedding_field` | `str`         | Name of the embedding field (default: "embedding") | `'embedding'` |
| `query_tensor`    | `str`         | Name of the query tensor (default: "q")            | `'q'`         |

Returns:

| Name          | Type          | Description                                        |
| ------------- | ------------- | -------------------------------------------------- |
| `RankProfile` | `RankProfile` | A Vespa RankProfile configured for semantic search |

Example

> > > config = ModelConfig(model_id="e5-small-v2", embedding_dim=384, binarized=False) profile = create_semantic_rank_profile(config) profile.name 'semantic'

### `create_hybrid_rank_profile(config, profile_name='fusion', base_profile='bm25', embedding_field='embedding', query_tensor='q', fusion_method='rrf', global_rerank_count=1000, first_phase_keep_rank_count=None)`

Create a hybrid ranking profile combining BM25 and semantic search.

Parameters:

| Name                          | Type            | Description                                                                                                                                                      | Default       |
| ----------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `config`                      | `ModelConfig`   | ModelConfig instance with model parameters                                                                                                                       | *required*    |
| `profile_name`                | `str`           | Name of the rank profile (default: "fusion")                                                                                                                     | `'fusion'`    |
| `base_profile`                | `str`           | Name of the BM25 profile to inherit from (default: "bm25")                                                                                                       | `'bm25'`      |
| `embedding_field`             | `str`           | Name of the embedding field (default: "embedding")                                                                                                               | `'embedding'` |
| `query_tensor`                | `str`           | Name of the query tensor (default: "q")                                                                                                                          | `'q'`         |
| `fusion_method`               | `FusionMethod`  | Fusion method - "rrf" for reciprocal rank fusion, "atan_norm" for atan-normalized sum in first phase, or "norm_linear" for linear normalization in global phase. | `'rrf'`       |
| `global_rerank_count`         | `int`           | Number of hits to rerank in global phase (default: 1000)                                                                                                         | `1000`        |
| `first_phase_keep_rank_count` | `Optional[int]` | How many documents to keep the first phase top rank values for (default: None, uses Vespa default of 10000)                                                      | `None`        |

Returns:

| Name          | Type          | Description                                      |
| ------------- | ------------- | ------------------------------------------------ |
| `RankProfile` | `RankProfile` | A Vespa RankProfile configured for hybrid search |

Example

> > > config = ModelConfig(model_id="e5-small-v2", embedding_dim=384) profile = create_hybrid_rank_profile(config) profile.name 'fusion'

### `get_model_config(model_name)`

Get a predefined model configuration by name.

Parameters:

| Name         | Type  | Description                | Default    |
| ------------ | ----- | -------------------------- | ---------- |
| `model_name` | `str` | Name of a predefined model | *required* |

Returns:

| Name          | Type          | Description             |
| ------------- | ------------- | ----------------------- |
| `ModelConfig` | `ModelConfig` | The model configuration |

Raises:

| Type       | Description                    |
| ---------- | ------------------------------ |
| `KeyError` | If the model name is not found |

Example

> > > config = get_model_config("e5-small-v2") config.embedding_dim 384

### `list_models()`

List all available predefined model configurations.

Returns:

| Type        | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `List[str]` | List of model names that can be used with get_model_config() |

Example

> > > models = list_models() 'e5-small-v2' in models True 'nomic-ai-modernbert' in models True

### `create_hybrid_package(models, app_name='hybridapp', schema_name='doc', global_rerank_count=1000)`

Create a Vespa application package configured for hybrid search evaluation.

This function creates a complete Vespa application package with all necessary components, fields, and rank profiles for evaluation. It supports single or multiple embedding models, automatically handling naming conflicts by using model-specific field and component names.

Parameters:

| Name                  | Type                                                                        | Description                                                                                                                                                                        | Default       |
| --------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `models`              | `Union[str, ModelConfig, List[Union[str, ModelConfig]], List[ModelConfig]]` | Single model or list of models to configure. Each can be: - A string model name (e.g., "e5-small-v2") to use a predefined config - A ModelConfig instance for custom configuration | *required*    |
| `app_name`            | `str`                                                                       | Name of the application (default: "hybridapp")                                                                                                                                     | `'hybridapp'` |
| `schema_name`         | `str`                                                                       | Name of the schema (default: "doc")                                                                                                                                                | `'doc'`       |
| `global_rerank_count` | `int`                                                                       | Number of hits to rerank in global phase (default: 1000)                                                                                                                           | `1000`        |

Returns:

| Name                 | Type                                   | Description                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ApplicationPackage` | `ApplicationPackageWithQueryFunctions` | Configured Vespa application package with: - Components for each embedding model - Embedding fields for each model (named "embedding" for single model, "embedding\_{component_id}" for multiple models) - BM25 and semantic rank profiles for each model - Hybrid rank profiles (RRF, atan_norm, norm_linear) for each model - A match-only rank profile for baseline evaluation |

Raises:

| Type         | Description                                   |
| ------------ | --------------------------------------------- |
| `ValueError` | If models list is empty                       |
| `KeyError`   | If a model name is not found in COMMON_MODELS |

Example

> > > #### Single model by name
> > >
> > > package = create_hybrid_package("e5-small-v2") len(package.components) 1 package.schema.document.fields[2].name 'embedding'
> > >
> > > #### Single model with custom config
> > >
> > > config = ModelConfig(model_id="my-model", embedding_dim=512) package = create_hybrid_package(config) package.schema.document.fields[2].name 'embedding'
> > >
> > > #### Multiple models - creates separate fields and profiles for each
> > >
> > > package = create_hybrid_package(["e5-small-v2", "e5-base-v2"]) len(package.components) 2
> > >
> > > #### Fields will be named: embedding_e5_small_v2, embedding_e5_base_v2
> > >
> > > field_names = [f.name for f in package.schema.document.fields if f.name.startswith('embedding')] len(field_names) 2
> > >
> > > #### Multiple models with mixed configs
> > >
> > > custom = ModelConfig(model_id="custom-model", embedding_dim=384) package = create_hybrid_package(["e5-small-v2", custom]) len(package.components) 2
