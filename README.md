<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# pyvespa

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

[![Documentation Status](https://readthedocs.org/projects/pyvespa/badge/?version=latest)](https://pyvespa.readthedocs.io/en/latest/?badge=latest)
<a href="https://cd.screwdriver.cd/pipelines/7055"><img src="https://cd.screwdriver.cd/pipelines/7055/badge"/></a>

[pyvespa site / documentation](https://pyvespa.readthedocs.io/en/latest/index.html)

[Vespa](https://vespa.ai/) is the scalable open-sourced serving engine that enables users to store,
compute and rank big data at user serving time.
`pyvespa` provides a python API to Vespa.
It allows users to create, modify, deploy and interact with running Vespa instances.
The main goal of the library is to allow for faster prototyping
and to facilitate Machine Learning experiments for Vespa applications -
also see [learntorank](https://github.com/vespa-engine/learntorank).


## License
Code licensed under the Apache 2.0 license. See [LICENSE](LICENSE) for terms.


## Development environment
Check [screwdriver.yaml](screwdriver.yaml) to see which packages and environment variables
need to be set to run unit and integration tests.


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
upload, this requires username and password:

```
python3 -m twine upload dist/*
```

At this point, the package has been released.
Create a new release tag at [github.com/vespa-engine/pyvespa/releases/new](https://github.com/vespa-engine/pyvespa/releases/new)
with a summary of the code changes.
