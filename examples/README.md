# NanoBEIR Evaluation Example

This example demonstrates how to use the `vespa.nanobeir` module to easily configure and run NanoBEIR evaluations with different embedding models.

## Overview

The `vespa.nanobeir` module provides utilities to simplify the creation of Vespa applications for information retrieval evaluation. It handles the complexity of configuring different embedding models with varying dimensions, tokenizers, and binary vs. float embeddings.

## Key Features

- **Model-centric configuration**: All model-specific parameters (dimension, tokenizer, binarization) are encapsulated in a `ModelConfig` object
- **Automatic field type selection**: The embedding field type is automatically set to `tensor<float>` or `tensor<int8>` based on whether embeddings are binarized
- **Automatic indexing configuration**: For binarized embeddings, `pack_bits` is automatically added to the indexing statement
- **Distance metric selection**: Uses hamming distance for binarized embeddings and cosine similarity (angular distance) for float embeddings
- **Predefined models**: Includes configurations for common models like e5-small-v2, e5-base-v2, snowflake-arctic-embed, and bge-m3

## Usage

### Basic Example

```python
from vespa.nanobeir import get_model_config, create_evaluation_package

# Get a predefined model configuration
config = get_model_config("e5-small-v2")

# Create a complete application package
package = create_evaluation_package(config, app_name="myeval")

# Deploy to Vespa Cloud or local Docker
# ... (deployment code)
```

### Custom Model Configuration

```python
from vespa.nanobeir import ModelConfig, create_embedder_component, create_embedding_field

# Define a custom model
config = ModelConfig(
    model_id="my-custom-model",
    embedding_dim=512,
    tokenizer_id="bert-base-uncased",
    binarized=False,
)

# Create individual components
embedder = create_embedder_component(config)
embedding_field = create_embedding_field(config)
```

### Binary Embeddings

```python
from vespa.nanobeir import ModelConfig

# Configure for binary embeddings
config = ModelConfig(
    model_id="bge-m3",
    embedding_dim=1024,  # Before packing
    binarized=True,
)

# The resulting field will be tensor<int8>(x[128]) with pack_bits in indexing
# The ranking profile will use hamming distance
```

## Running the Example

```bash
# From the repository root
uv run python examples/nanobeir_evaluation_example.py
```

This will demonstrate:
1. Creating packages for different float embedding models (e5-small-v2, e5-base-v2)
2. Creating a package for binary embeddings (bge-m3-binary)
3. Creating a package with custom model configuration
4. Listing all available predefined models

## Available Predefined Models

- `e5-small-v2`: 384-dimensional float embeddings
- `e5-base-v2`: 768-dimensional float embeddings
- `snowflake-arctic-embed-xs`: 384-dimensional float embeddings
- `snowflake-arctic-embed-s`: 384-dimensional float embeddings
- `snowflake-arctic-embed-m`: 768-dimensional float embeddings
- `bge-m3-binary`: 1024-dimensional binary embeddings (packed to 128 int8 values)

## Next Steps

After creating an application package:

1. **Deploy to Vespa**: Use `VespaCloud` or `VespaDocker` to deploy your application
2. **Feed documents**: Load the NanoBEIR dataset and feed documents to Vespa
3. **Run evaluation**: Use `VespaEvaluator` or `VespaMatchEvaluator` to evaluate retrieval quality
4. **Compare models**: Run the same evaluation with different model configurations to compare performance

## Related Documentation

- [vespa.nanobeir API Reference](../vespa/nanobeir.py)
- [vespa.evaluation API Reference](../vespa/evaluation.py)
- [Vespa Documentation - Embeddings](https://docs.vespa.ai/en/embedding.html)
- [Vespa Documentation - Binary Quantization](https://docs.vespa.ai/en/embedding.html#binary-quantization)
