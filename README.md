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
See [README](https://github.com/vespa-engine/pyvespa/tree/master/vespacli) and [Veso]

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


### Release instructions
* Check out master branch
* Temporarily change library version number in `get_target_version()` in [setup.py](setup.py) to the new version,
  e.g. "0.16.0".
* Run from the pyvespa root directory to create the library files:

```
python3 -m pip install --upgrade pip
python3 -m pip install twine wheel

python3 setup.py sdist bdist_wheel
``` 

With write access to [pypi.org/project/pyvespa/](https://pypi.org/project/pyvespa/),
upload, this requires username "__token__" and the token value as password, including the pypi- prefix:

```
python3 -m twine upload dist/*
```

At this point, the package has been released.
Create a new release tag at [github.com/vespa-engine/pyvespa/releases/new](https://github.com/vespa-engine/pyvespa/releases/new)
with a summary of the code changes.
