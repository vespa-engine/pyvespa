# Project Overview

This project is **pyvespa**, a Python API for [Vespa.ai](https://vespa.ai/) - the scalable open-source search engine. pyvespa enables users to create, modify, deploy, and interact with Vespa applications directly from Python. The main goal is to allow for faster prototyping and provide an accessible way to leverage Vespa's advanced search capabilities including vector search, hybrid retrieval, ranking, and real-time serving.

## Repository Structure

- `/vespa/`: Core Python package containing the main API modules
  - `/application.py`: Main Vespa application interface for data-plane operations, such as queries, feeding, and visiting
  - `/package.py`: Application package definitions, schemas, fields, and configurations
  - `/deployment.py`: Deployment and control-plane interfaces for Vespa Cloud and Docker
  - `/evaluation.py`: Tools for evaluating search quality and performance
  - `/io.py`: Response handling and data structures
  - `/querybuilder/`: Python DSL for building YQL queries
  - `/configuration/`: XML configuration generation and services setup
- `/tests/`: Comprehensive test suite with unit and integration tests
  - `/unit/`: Unit tests for individual components
  - `/integration/`: Integration tests with actual Vespa deployments
- `/docs/`: Documentation and example notebooks
  - `/sphinx/source/`: Sphinx documentation source and Jupyter notebooks
  - `/examples/`: Practical examples and tutorials (in the format of Jupyter notebooks)
- `/vespacli/`: Python package wrapper of Vespa CLI (This should generally not be modified)
- Root-level scripts: `feed_to_vespa.py`, `feed-split.py` for splitting and feeding data to the Vespa Docsearch application.

## Core Dependencies and Libraries

**Primary Dependencies:**
- `requests` and `httpx` - HTTP client libraries for API communication
- `docker` - Docker container management for local deployments
- `jinja2` - Template engine for configuration file generation
- `cryptography` - Certificate and key management for Vespa Cloud
- `lxml` - XML processing and validation
- `fastcore` - Utility functions and decorators
- `tenacity` - Retry logic for robust networking

**Development Dependencies:**
- `pytest` and `unittest` - Testing frameworks
- `datasets` - For loading and processing data in examples
- Various ML libraries in examples: `transformers`, `torch`, `lightgbm`, `pandas`

## Coding Standards and Conventions

**General Guidelines:**
- Strive for simplicity and clarity in code
- Avoid adding external dependencies unless absolutely necessary and approved by user.

**Python Standards:**
- Use Python 3.9+ features and type hints throughout
- Follow PEP 8 style guidelines with meaningful variable names
- Use lowercase with underscores for function and variable names (`snake_case`)
- Use PascalCase for class names
- Include comprehensive docstrings for all public methods and classes

**API Design Patterns:**
- Use context managers (`with app.syncio()`, `async with app.asyncio()`) for connection pooling
- Provide both synchronous and asynchronous interfaces
- Use dataclasses and TypedDict for structured configuration
- Follow builder pattern for complex objects (ApplicationPackage, Schema, etc.)
- Use callback functions for handling feed operations and responses

**Error Handling:**
- Raise `VespaError` for Vespa-specific exceptions
- Use tenacity decorators for automatic retry logic
- Validate inputs early and provide clear error messages
- Check response status codes and provide meaningful feedback

**Testing Practices:**
- Write both unit tests (using unittest/pytest) and integration tests
- Use mock objects for external dependencies
- Test both successful operations and error conditions
- Include doctests in example code within notebooks

**Configuration and Templates:**
- Use Jinja2 templates for generating XML configurations
- Validate XML against RelaxNG schemas where available
- Support both programmatic and declarative configuration styles
- Maintain backward compatibility when possible

## Key Architecture Patterns

**Application Package Creation:**
```python
from vespa.package import ApplicationPackage, Schema, Document, Field, RankProfile

app_package = ApplicationPackage(name="myapp")
app_package.schema.add_fields(
    Field(name="title", type="string", indexing=["index", "summary"])
)
```

**Deployment Patterns:**
- Local Docker deployment via `VespaDocker`
- Vespa Cloud deployment via `VespaCloud` with certificate authentication
- Support for both development and production configurations

**Query and Feed Operations:**
- Use connection pooling and compression for performance
- Support batch operations and iterables for large datasets
- Provide progress callbacks for long-running operations
- Handle rate limiting and retries automatically

**Example Code Structure:**
- Start with package installation and imports
- Create application package with schema definition
- Deploy to target environment (Docker or Cloud)
- Feed data using iterables or individual documents
- Execute queries with various ranking strategies
- Include cleanup and teardown procedures

## Documentation source

Very often you need to consult the Vespa documentation. For example for format of the XML configuration files, or for documentation of a specific query parameter, and so on. Find relevant info using the context7 MCP server that you have available. (repo: vespa-engine/documentation)


## Development Workflow

We use `uv` for managing dependencies.
To install the project with development dependencies, run:
```bash
uv sync --extra dev
``` 

After that you can run tests with:
```bash
uv run pytest tests/unit/ -v
```
You can also replace `unit`with `integration` to run integration tests, or just `tests/` to run all tests, or run specific test files or test cases the same way.
