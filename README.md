<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# pyvespa

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

[![Documentation Status](https://readthedocs.org/projects/pyvespa/badge/?version=latest)](https://pyvespa.readthedocs.io/en/latest/?badge=latest)

![GitHub Release](https://img.shields.io/github/v/release/vespa-engine/pyvespa)
![PyPI - Version](https://img.shields.io/pypi/v/pyvespa)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyvespa)


[pyvespa site / documentation](https://pyvespa.readthedocs.io/en/latest/index.html)

## Overview

[Vespa](https://vespa.ai/) is the scalable open-sourced serving engine that enables users to store,
compute and rank big data at user serving time.
`pyvespa` provides a Python API to Vespa.
It allows users to create, modify, deploy and interact with running Vespa instances.
The main goal of the library is to allow for faster prototyping and get familiar with Vespa features.

## Installation Steps

### Using Package Manager

Install directly from PyPI:

```bash
pip install pyvespa
```

### Verification

To verify the successful installation, run the following command:

```python
python -m pip show pyvespa
```

### Additional Information
Please refer to the [official document](https://pyvespa.readthedocs.io/en/stable/) for more installation details.

## Development Environment

To install an editable version of the library with development dependencies, run the following command from the root directory of the repository:

```python
pip install -e ".[dev]"
```

Note that this will enforce linting and formatting with [Ruff](https://github.com/astral-sh/ruff), which will also be triggered by a [pre-commit](https://pre-commit.com/) hook.

This means you may encounter an error message when trying to commit changes if the code does not pass the linting and formatting checks. The errors are detailed in the output, and you can optionally run them manually with the `ruff` CLI tool.

## Releases

Find releases and release notes on [GitHub](https://github.com/vespa-engine/pyvespa/releases).

### Release Details

The release flow is semi-automated but involves a few manual steps:

1. Create a new release from [github.com/vespa-engine/pyvespa/releases/new](https://github.com/vespa-engine/pyvespa/releases/new).
2. Tag the release with the version number, e.g., `v0.41.0`.
3. This tag will trigger a GitHub Action that will publish the package to [PyPI](https://pypi.org/project/pyvespa/).
4. A PR will also be automatically created to update the affected files with the new version. Merge this PR to keep the repository's version updated.

This workflow can also be dispatched manually, but note that steps 3 and 4 will ONLY be triggered by a release.

## License

Code is licensed under the Apache 2.0 license. See [LICENSE](LICENSE) for terms.

## vespacli

This repo also contains the python wrapper for the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli).
See [README](https://github.com/vespa-engine/pyvespa/tree/master/vespacli).

## External Documents

For more details, refer to additional documents:

- [Documentation](https://pyvespa.readthedocs.io/en/latest/index.html)
- [Vespa CLI README](https://github.com/vespa-engine/pyvespa/tree/master/vespacli)

## Version History

For version history, refer to the [GitHub Releases](https://github.com/vespa-engine/pyvespa/releases).

## Help and Support

For FAQs, commonly encountered errors, or further assistance, please refer to:

- [Pyvespa Documentation](https://pyvespa.readthedocs.io/en/latest/index.html)
- Report issues on the [GitHub Issues Page](https://github.com/vespa-engine/pyvespa/issues).
