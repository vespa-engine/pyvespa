# Vespa python API

[Vespa](https://vespa.ai/) is the scalable open-sourced serving engine to store, compute and rank big data at user serving time. `pyvespa` provides a python API to Vespa - use it to create, modify, deploy and interact with running Vespa instances. The main goal of the library is to allow for faster prototyping and get familiar with Vespa features.

!!!warning
    pyvespa is under active development and backward incompatible changes may occur.

[Hybrid Search](getting-started-pyvespa.ipynb) - Quickstart is a good primer on how to create an application, feed data and run queries. See Examples for use cases. The following blog post series will get you started:

* [Run search engine experiments in Vespa from python](https://blog.vespa.ai/run-search-engine-experiments-in-Vespa-from-python/)

* [Build sentence/paragraph level QA application from python with Vespa](https://blog.vespa.ai/build-qa-app-from-python-with-vespa/)

* [Build a basic text search application from python with Vespa: Part 1](https://blog.vespa.ai/build-basic-text-search-app-from-python-with-vespa/)

* [Build a News recommendation app from python with Vespa: Part 1](https://blog.vespa.ai/build-news-search-app-from-python-with-vespa/)

The [Vespa FAQ](https://docs.vespa.ai/en/faq.html) is a great resource, also see pyvespa troubleshooting.

## Requirements

Install ``pyvespa``:

```bash
  python3 -m pip install pyvespa
```

Install [jupyter notebook](https://jupyter.org/install#jupyter-notebook) to run the notebooks in a browser:

```bash
  git clone --depth 1 https://github.com/vespa-engine/pyvespa.git
  jupyter notebook --notebook-dir pyvespa/docs/sphinx/source
```
Many of the pyvespa guides / notebooks use Docker -
minimum memory requirement is 4 Gb unless other documented:

```bash
  docker info | grep "Total Memory"
  or
  podman info | grep "memTotal"
```

One can also use [Vespa Cloud](getting-started-pyvespa-cloud) to run the notebooks.