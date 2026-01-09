# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyvespa is a Python API for Vespa.ai, the scalable open-source serving engine. It enables users to create, modify, deploy, and interact with Vespa applications from Python. The library facilitates rapid prototyping and provides access to Vespa's advanced features including vector search, hybrid retrieval, ranking, and real-time serving.

The repository also contains `vespacli`, a Python wrapper for the Vespa CLI tool (generally should not be modified).

## Development Setup

Install the development environment using `uv`:

```bash
uv sync --extra dev
```

This enforces linting and formatting with Ruff via pre-commit hooks. Commits will fail if code doesn't pass linting checks.

## Common Commands

### Testing

```bash
# Run unit tests
uv run pytest tests/unit/ -v

# Run integration tests (requires Docker/Vespa)
uv run pytest tests/integration/ -v

# Run specific test file
uv run pytest tests/unit/test_package.py -v

# Run specific test
uv run pytest tests/unit/test_package.py::TestField::test_field_name -v

# Run doctests in documentation
uv sync --extra dev --extra docs
uv run pytest tests/mktestdocs -s -v

# Run performance tests (marked with @pytest.mark.perf)
uv run pytest tests/perf/ -m perf -v
```

### Linting and Formatting

```bash
# Run Ruff linter with auto-fix
uv run ruff check --fix

# Run Ruff formatter
uv run ruff format

# Pre-commit will automatically run these on commit
```

### Building and Installation

```bash
# Install in editable mode with dev dependencies
uv sync --extra dev

# Build the package
uv build

# Install additional dependency groups
uv sync --extra notebooks  # For Jupyter notebook examples
uv sync --extra docs       # For documentation generation
uv sync --extra feed       # For data feeding scripts
```

## Architecture Overview

### Core Module Structure

The `/vespa/` directory contains the main API:

- **`application.py`**: Data-plane operations - queries, feeding documents, visiting collections, connection management. Main entry point for interacting with deployed Vespa instances.

- **`package.py`**: Application package definitions - schemas, fields, document types, rank profiles, query profiles. The builder API for defining Vespa applications programmatically.

- **`deployment.py`**: Control-plane operations - deploying to Vespa Cloud (`VespaCloud`) or Docker (`VespaDocker`), managing certificates, environment configuration.

- **`io.py`**: Response objects (`VespaResponse`, `VespaQueryResponse`, `VespaVisitResponse`) that wrap HTTP responses with convenient accessors for common data.

- **`querybuilder/`**: Python DSL for building YQL (Vespa Query Language) queries programmatically instead of raw strings.

- **`configuration/`**: XML configuration generation using Jinja2 templates for services.xml, deployment.xml, and query profiles.

- **`evaluation.py`**: Tools for evaluating search quality, and measuring performance metrics.

- **`models.py`**: Data models and type definitions used across the codebase.

- **`throttling.py`**: Adaptive throttling for feed operations to prevent overwhelming Vespa instances.

### Key Architectural Patterns

**1. Connection Management**

The library uses context managers for connection pooling:

```python
# Synchronous
with app.syncio() as session:
    response = app.query(body=query_body, session=session)

# Asynchronous
async with app.asyncio() as session:
    response = await app.query_async(body=query_body, session=session)
```

Sessions should be reused across multiple operations for optimal performance through connection pooling.

**2. Builder Pattern for Application Packages**

Application packages are constructed programmatically using builder pattern:

```python
from vespa.package import ApplicationPackage, Schema, Document, Field, RankProfile

app_package = ApplicationPackage(name="myapp")
app_package.schema.add_fields(
    Field(name="title", type="string", indexing=["index", "summary"]),
    Field(name="embedding", type="tensor<float>(x[384])", indexing=["attribute"])
)
```

**3. HTTP Client Migration**

The codebase is migrating from `requests`/`httpx` to `httpr` (a Rust-based HTTP client). You may see temporary comparison code - the target is to use `httpr` exclusively for production operations.

**4. Dual Sync/Async APIs**

Most operations provide both synchronous and asynchronous interfaces:
- Sync: `app.query()`, `app.feed_data_point()`, `app.delete_data()`
- Async: `app.query_async()`, `app.feed_data_point_async()`, `app.delete_data_async()`

**5. Configuration via Templates**

XML configuration files (services.xml, deployment.xml) are generated from Python objects using Jinja2 templates in `/vespa/templates/`. The `vespa.configuration` module provides type-safe Python APIs that render to XML.

**6. Feed Operations with Callbacks**

Feeding operations support callbacks for progress tracking and error handling:

```python
def callback(response: VespaResponse, doc_id: str):
    if not response.is_successful():
        print(f"Failed to feed {doc_id}")

app.feed_iterable(documents, callback=callback)
```

## Important Patterns and Conventions

### Response Handling

All HTTP operations return response objects that check for errors automatically. The `raise_for_status()` function in `application.py` wraps HTTP errors with `VespaError` for better error messages.

### Type Hints

The codebase uses comprehensive type hints. Use Python 3.9+ type syntax and `typing_extensions` for backports (e.g., `Unpack` for Python < 3.11).

### Tenacity for Retries

Network operations use `@retry` decorators from `tenacity` for automatic retry with exponential backoff. See existing patterns in `application.py` and `deployment.py`.

### XML Validation

When modifying XML configuration generation, validate against RelaxNG schemas in `/vespa/configuration/relaxng/` if available.

### Test Structure

- **Unit tests** (`tests/unit/`): Mock external dependencies, test individual components in isolation
- **Integration tests** (`tests/integration/`): Spin up Docker containers with real Vespa instances
- **Performance tests** (`tests/perf/`): Load testing marked with `@pytest.mark.perf`
- **Doctest tests** (`tests/mktestdocs/`): Validate code examples in documentation

## Coding Standards

- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Line length**: 88 characters (enforced by Ruff)
- **Docstrings**: Required for all public methods and classes
- **Error handling**: Raise `VespaError` for Vespa-specific errors
- **Imports**: Standard library, third-party, then local imports
- **Type hints**: Required for function signatures and class attributes
- **Avoid adding dependencies**: Consult maintainers before adding new external dependencies

## Documentation Resources

For Vespa-specific documentation (e.g., XML configuration formats, query parameters, YQL syntax), use the context7 MCP server to search the `vespa-engine/documentation` repository.

For library-specific documentation (`httpx`, `docker`, `jinja2`, etc.), use context7 MCP server to search their respective documentation.

## Release Process

Releases are semi-automated:
1. Create a new release from GitHub with version tag (e.g., `v0.41.0`)
2. GitHub Actions automatically publishes to PyPI
3. A PR is auto-created to update version files in the repository
4. Merge the version update PR to keep the repo synchronized

## Notes

- The main branch is `master` (not `main`)
- HTTP client libraries: Currently transitioning from `requests`/`httpx` to `httpr`
- Vespa Cloud operations require certificate authentication handled by `VespaCloud` class
- Docker deployments use the `docker` Python client to manage local containers
- Query profiles and deployment configurations use XML under the hood but are defined via Python objects
