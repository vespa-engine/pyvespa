# pyvespa

[![Documentation Status](https://readthedocs.org/projects/pyvespa/badge/?version=latest)](https://pyvespa.readthedocs.io/en/latest/?badge=latest)
<a href="https://cd.screwdriver.cd/pipelines/7055"><img src="https://cd.screwdriver.cd/pipelines/7055/badge"/></a>

[Vespa](https://vespa.ai/) is the scalable open-sourced serving engine that enable us to store, compute and rank big data at user serving time. `pyvespa` provides a python API to Vespa. It allow us to create, modify, deploy and interact with running Vespa instances. The main goal of the library is to allow for faster prototyping and to facilitate Machine Learning experiments for Vespa applications.

* [pyvespa official documentation](https://pyvespa.readthedocs.io/en/latest/index.html) 

## License

Code licensed under the Apache 2.0 license. See [LICENSE](LICENSE) for terms.

## Development setup environment

Check the file `screwdriver.yaml` to see which packages and environment variables
need to be set to run unit and integration tests.

## Code format

This repo uses the default configuration of [Black](https://github.com/psf/black) to 
standardize code formatting. 

## Release instructions

We currently release new pyvespa versions manually.

* Check out master branch
* Manually change library version number on setup.py from `get_target_version()` to the desired version, e.g. "0.16.0".
* Run the following command from the pyvespa root directory to create the library files

```
python3 -m pip install --upgrade pip
python3 -m pip install twine wheel

python3 setup.py sdist bdist_wheel
``` 

* Make sure you have the python library `twine` installed and access to the pyvespa project on PyPI. Run the following command to send the library to PyPI, it requires your username and password:

```
python3 -m twine upload dist/*
```

At this point, the latest package has been released. 

* Please, create a new release tag on the Github project (https://github.com/vespa-engine/pyvespa/releases/new) with a summary of the code changes included in the release. Look at previous release notes and follow the same pattern.