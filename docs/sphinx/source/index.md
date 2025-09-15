# Vespa python API

[Vespa](https://vespa.ai/) is the scalable open-sourced serving engine to store, compute and rank big data at user serving time. `pyvespa` provides a python API to Vespa.

We aim for complete feature parity with Vespa, and estimate that we cover > 95% of Vespa features, with all most commonly used features supported.

If you find a Vespa feature that you are not able to express/use with `pyvespa`, please [open an issue](https://github.com/vespa-engine/pyvespa/issues/new/choose).

## Quick start

To get a sense of the most basic functionality, check out the Hybrid Search Quick start: 

- [Hybrid search quick start - Docker](getting-started-pyvespa)
- [Hybrid search quick start - Vespa Cloud](/getting-started-pyvespa-cloud)

## Overview of pyvespa features

!!! info 
    There are two main interfaces to Vespa:

    1. Control-plane API: Used to deploy and manage Vespa applications.
        - [`VespaCloud`](/api/vespa/deployment.html#vespa.deployment.VespaCloud): Control-plane interface to Vespa Cloud.
        - [`VespaDocker`](/api/vespa/deployment.html#vespa.deployment.VespaDocker): Control-plane iterface to local Vespa instance (docker/podman).
    2. Data-plane API: Used to feed and query data in Vespa applications.
        -  [`Vespa`](/api/vespa/application.html#vespa.application.Vespa)

    Note that `VespaCloud` and `Vespa` require two separate authentication methods.

    Refer to the [Authenticating to Vespa Cloud](/authenticating-to-vespa-cloud) for details.

- Create and deploy application packages, including schemas, rank profiles, `services.xml`, query profiles etc.
- [Feed and retrieve](/reads-writes) documents to/from Vespa, using `/document/v1/` API.
- [Query](/query) Vespa applications, using `/search/` API.
- [Build complex queries](/query.html#using-the-querybuilder-dsl-api) using the [`QueryBuilder`](/api/vespa/querybuilder/builder/builder.html) API.
- [Collect training data](/evaluating-vespa-application-cloud) for ML using [`VespaFeatureCollector`](/api/vespa/evaluation.html#vespa.evaluation.VespaFeatureCollector).
- [Evaluate](/evaluating-vespa-application-cloud) Vespa applications using [`VespaEvaluator`](/api/vespa/evaluation.html#vespa.evaluation.VespaEvaluator)/[`VespaMatchEvaluator`](/api/vespa/evaluation.html#vespa.evaluation.VespaMatchEvaluator).

## Requirements

Install `pyvespa`:

We recommend using [`uv`](https://docs.astral.sh/uv/) to manage your python environments:

    uv add pyvespa

or using `pip`:

    pip install pyvespa

## Check out the examples

Check out our wide variety of [Examples](/examples/) that demonstrate how to use the Vespa Python API to serve various use cases.