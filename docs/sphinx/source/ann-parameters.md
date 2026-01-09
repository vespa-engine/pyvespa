# ANN Parameter Tuning

Approximate Nearest Neighbor (ANN) search is a powerful way to make vector search scalable and efficient. In Vespa, this is implemented by building HNSW graphs for embedding fields.

For a search that uses _only_ vector similarity for retrieval, this works very well as you can just query the HNSW index and get (enough) relevant results back very fast.
However, most Vespa applications are more complex and often combine vector similarity with filtering on metadata fields.

There are multiple strategies in Vespa for handling queries that combine ANN with filtering,
and there are parameters that control the strategy selection and the strategies themselves.
While Vespa has chosen default values for these parameters that work well in most use cases, one often can benefit from further tuning these parameters for the application/use case/data set at hand.

## ANN Parameter Optimizer

The `vespa.evaluation` module provides a `VespaNNParameterOptimizer` class that, given a sufficient sample of queries
using ANN with filtering,
performs measurements to analyze the effect of various tuning parameters and, based on this,
provides suggestions for these parameters.
Running the optimizer can be as simple as this:

```python
from vespa.evaluation import VespaNNParameterOptimizer

optimizer = VespaNNParameterOptimizer(
    app=my_vespa_app,
    queries=my_list_of_queries,
    hits=number_of_target_hits_used_in_my_queries,
)
report = optimizer.run()

suggested_parameters = {
    "ranking.matching.approximateThreshold": report["approximateThreshold"][
        "suggestion"
    ],
    "ranking.matching.filterFirstThreshold": report["filterFirstThreshold"][
        "suggestion"
    ],
    "ranking.matching.filterFirstExploration": report["filterFirstExploration"][
        "suggestion"
    ],
    "ranking.matching.postFilterThreshold": report["postFilterThreshold"][
        "suggestion"
    ],
}
```

See the
[example](https://vespa-engine.github.io/pyvespa/examples/ann-parameter-tuning-vespa-cloud.html)
for a full guide on how to use this class and how to interpret the report it produces.
See the
[documentation](https://vespa-engine.github.io/pyvespa/api/vespa/evaluation.html#vespa.evaluation.VespaNNParameterOptimizer)
for further details.

