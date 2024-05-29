<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# pyvespa

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://vespa.ai/assets/vespa-ai-logo-heather.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://vespa.ai/assets/vespa-ai-logo-rock.svg">
  <img alt="#Vespa" width="200" src="https://vespa.ai/assets/vespa-ai-logo-rock.svg" style="margin-bottom: 25px;">
</picture>

[![Documentation Status](https://readthedocs.org/projects/pyvespa/badge/?version=latest)](https://pyvespa.readthedocs.io/en/latest/?badge=latest)
<a href="https://cd.screwdriver.cd/pipelines/7055"><img src="https://cd.screwdriver.cd/pipelines/7055/badge"/></a>

[pyvespa site / documentation](https://pyvespa.readthedocs.io/en/latest/index.html)

[Vespa](https://vespa.ai/) is the scalable open-sourced serving engine that enables users to store,
compute and rank big data at user serving time.
`pyvespa` provides a python API to Vespa.
It allows users to create, modify, deploy and interact with running Vespa instances.
The main goal of the library is to allow for faster prototyping and get familiar with Vespa features.

## vespacli

This repo also contains the python wrapper for the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli).
See [README](https://github.com/vespa-engine/pyvespa/tree/master/vespacli).

## License

Code licensed under the Apache 2.0 license. See [LICENSE](LICENSE) for terms.

## Development environment

To install editable version of the library with dev dependencies, run the following command from the root directory of the repository:

```python
pip install -e ".[dev]"
```

Note that this will enforce linting and formatting with [Ruff](https://github.com/astral-sh/ruff), which also will be triggered by a [pre-commit](https://pre-commit.com/)-hook.

This means that you may get an error message when trying to commit changes if the code does not pass the linting and formatting checks. The errors are detailed in the output, and you can optionally run manually with `ruff` CLI-tool.

## Releases

Find releases and release notes on [GitHub](https://github.com/vespa-engine/pyvespa/releases).

### Release details

The release flow is semi-automated, but involves a few manual steps.

1. Create a new release from [github.com/vespa-engine/pyvespa/releases/new](https://github.com/vespa-engine/pyvespa/releases/new).
2. Make sure to tag the release with the version number, e.g., `v0.41.0`.
3. This tag will trigger a github action that will publish the package to [PyPI](https://pypi.org/project/pyvespa/).
4. A PR will also be automatically created to update the affected files with the new version. This PR should be merged to keep the version updated in the repository.

This workflow can also be dispatched manually, but note that steps 3 and 4 will ONLY be triggered by a release.
