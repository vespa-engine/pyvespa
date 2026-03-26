# Vespa python API

[Vespa](https://vespa.ai/) is the scalable open-sourced serving engine to store, compute and rank big data at user serving time. `pyvespa` provides a python API to Vespa.

We aim for complete feature parity with Vespa, and estimate that we cover > 95% of Vespa features, with all most commonly used features supported.

If you find a Vespa feature that you are not able to express/use with `pyvespa`, please [open an issue](https://github.com/vespa-engine/pyvespa/issues/new/choose).

## Quick start

To get a sense of the most basic functionality, check out the Hybrid Search Quick start:

- [Hybrid search quick start - Docker](https://vespa-engine.github.io/pyvespa/getting-started-pyvespa.md)
- [Hybrid search quick start - Vespa Cloud](https://vespa-engine.github.io/pyvespa/getting-started-pyvespa-cloud.md)

## Overview of pyvespa features

Info

There are two main interfaces to Vespa:

1. Control-plane API: Used to deploy and manage Vespa applications.
   - [`VespaCloud`](https://vespa-engine.github.io/pyvespa/api/vespa/deployment.md#vespa.deployment.VespaCloud): Control-plane interface to Vespa Cloud.
   - [`VespaDocker`](https://vespa-engine.github.io/pyvespa/api/vespa/deployment.md#vespa.deployment.VespaDocker): Control-plane iterface to local Vespa instance (docker/podman).
1. Data-plane API: Used to feed and query data in Vespa applications.
   - [`Vespa`](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa)

Note that `VespaCloud` and `Vespa` require two separate authentication methods.

Refer to the [Authenticating to Vespa Cloud](https://vespa-engine.github.io/pyvespa/authenticating-to-vespa-cloud.md) for details.

- Create and deploy application packages, including schemas, rank profiles, `services.xml`, query profiles etc.
- [Feed and retrieve](https://vespa-engine.github.io/pyvespa/reads-writes.md) documents to/from Vespa, using `/document/v1/` API.
- [Query](https://vespa-engine.github.io/pyvespa/query.md) Vespa applications, using `/search/` API.
- [Build complex queries](https://vespa-engine.github.io/pyvespa/query.md#using-the-querybuilder-dsl-api) using the [`QueryBuilder`](https://vespa-engine.github.io/pyvespa/api/vespa/querybuilder/builder/builder.md) API.
- [Collect training data](https://vespa-engine.github.io/pyvespa/evaluating-vespa-application-cloud.md) for ML using [`VespaFeatureCollector`](https://vespa-engine.github.io/pyvespa/api/vespa/evaluation.md#vespa.evaluation.VespaFeatureCollector).
- [Evaluate](https://vespa-engine.github.io/pyvespa/evaluating-vespa-application-cloud.md) Vespa applications using [`VespaEvaluator`](https://vespa-engine.github.io/pyvespa/api/vespa/evaluation.md#vespa.evaluation.VespaEvaluator)/[`VespaMatchEvaluator`](https://vespa-engine.github.io/pyvespa/api/vespa/evaluation.md#vespa.evaluation.VespaMatchEvaluator).

## Requirements

Install `pyvespa`:

We recommend using [`uv`](https://docs.astral.sh/uv/) to manage your python environments:

```text
uv add pyvespa
```

or using `pip`:

```text
pip install pyvespa
```

## Check out the examples

Check out our wide variety of [Examples](https://vespa-engine.github.io/pyvespa/examples/index.md) that demonstrate how to use the Vespa Python API to serve various use cases.
