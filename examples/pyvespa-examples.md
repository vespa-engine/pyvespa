# Pyvespa examples[¶](#pyvespa-examples)

This is a notebook with short examples one can build applications from.

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md), which also has utilies for debugging.

In \[ \]:

Copied!

```
!pip3 install pyvespa
```

!pip3 install pyvespa

## Neighbors[¶](#neighbors)

Explore distance between points in 3D vector space.

These are simple examples, feeding documents with a tensor representing a point in space, and a rank profile calculating the distance between a point in the query and the point in the documents.

The examples start with using simple ranking expressions like [euclidean-distance](https://docs.vespa.ai/en/reference/ranking-expressions.html#euclidean-distance-t), then rank features like [closeness()](<https://docs.vespa.ai/en/reference/rank-features.html#closeness(dimension,name)>) and setting different [distance-metrics](https://docs.vespa.ai/en/nearest-neighbor-search.html#distance-metrics-for-nearest-neighbor-search).

### Distant neighbor[¶](#distant-neighbor)

First, find the point that is **most** distant from a point in query - deploy the Application Package:

In \[14\]:

Copied!

```
from vespa.package import ApplicationPackage, Field, RankProfile
from vespa.deployment import VespaDocker
from vespa.io import VespaResponse

app_package = ApplicationPackage(name="neighbors")

app_package.schema.add_fields(
    Field(name="point", type="tensor<float>(d[3])", indexing=["attribute", "summary"])
)

app_package.schema.add_rank_profile(
    RankProfile(
        name="max_distance",
        inputs=[("query(qpoint)", "tensor<float>(d[3])")],
        first_phase="euclidean_distance(attribute(point), query(qpoint), d)",
    )
)

vespa_docker = VespaDocker()
app = vespa_docker.deploy(application_package=app_package)
```

from vespa.package import ApplicationPackage, Field, RankProfile from vespa.deployment import VespaDocker from vespa.io import VespaResponse app_package = ApplicationPackage(name="neighbors") app_package.schema.add_fields( Field(name="point", type="tensor<float>(d[3])", indexing=["attribute", "summary"]) ) app_package.schema.add_rank_profile( RankProfile( name="max_distance", inputs=\[("query(qpoint)", "tensor<float>(d[3])")\], first_phase="euclidean_distance(attribute(point), query(qpoint), d)", ) ) vespa_docker = VespaDocker() app = vespa_docker.deploy(application_package=app_package)

```
Waiting for configuration server, 0/300 seconds...
Waiting for configuration server, 5/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 0/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 5/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 10/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 15/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 20/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 25/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Application is up!
Finished deployment.
```

Feed points in 3d space using a 3-dimensional [indexed tensor](https://docs.vespa.ai/en/tensor-user-guide.html). Pyvespa feeds using the [/document/v1/ API](https://docs.vespa.ai/en/reference/document-v1-api-reference.html), refer to [document format](https://docs.vespa.ai/en/reference/document-json-format.html):

In \[15\]:

Copied!

```
def get_feed(field_name):
    return [
        {"id": 0, "fields": {field_name: [0.0, 1.0, 2.0]}},
        {"id": 1, "fields": {field_name: [1.0, 2.0, 3.0]}},
        {"id": 2, "fields": {field_name: [2.0, 3.0, 4.0]}},
    ]


with app.syncio(connections=1) as session:
    for u in get_feed("point"):
        response: VespaResponse = session.update_data(
            data_id=u["id"], schema="neighbors", fields=u["fields"], create=True
        )
        if not response.is_successful():
            print(
                "Update failed for document {}".format(u["id"])
                + " with status code {}".format(response.status_code)
                + " with response {}".format(response.get_json())
            )
```

def get_feed(field_name): return \[ {"id": 0, "fields": {field_name: [0.0, 1.0, 2.0]}}, {"id": 1, "fields": {field_name: [1.0, 2.0, 3.0]}}, {"id": 2, "fields": {field_name: [2.0, 3.0, 4.0]}}, \] with app.syncio(connections=1) as session: for u in get_feed("point"): response: VespaResponse = session.update_data( data_id=u["id"], schema="neighbors", fields=u["fields"], create=True ) if not response.is_successful(): print( "Update failed for document {}".format(u["id"])

- " with status code {}".format(response.status_code)
- " with response {}".format(response.get_json()) )

**Note:** The feed above uses [create-if-nonexistent](https://docs.vespa.ai/en/document-v1-api-guide.html#create-if-nonexistent), i.e. update a document, create it if it does not exists. Later in this notebook we will add a field and update it, so using an update to feed data makes it easier.

Query from origo using [YQL](https://docs.vespa.ai/en/query-language.html). The rank profile will rank the most distant points highest, here `sqrt(2*2 + 3*3 + 4*4) = 5.385`:

In \[16\]:

Copied!

```
import json
from vespa.io import VespaQueryResponse

result: VespaQueryResponse = app.query(
    body={
        "yql": "select point from neighbors where true",
        "input.query(qpoint)": "[0.0, 0.0, 0.0]",
        "ranking.profile": "max_distance",
        "presentation.format.tensors": "short-value",
    }
)

if not response.is_successful():
    print(
        "Query failed with status code {}".format(response.status_code)
        + " with response {}".format(response.get_json())
    )
    raise Exception("Query failed")
if len(result.hits) != 3:
    raise Exception("Expected 3 hits, got {}".format(len(result.hits)))
print(json.dumps(result.hits, indent=4))
```

import json from vespa.io import VespaQueryResponse result: VespaQueryResponse = app.query( body={ "yql": "select point from neighbors where true", "input.query(qpoint)": "[0.0, 0.0, 0.0]", "ranking.profile": "max_distance", "presentation.format.tensors": "short-value", } ) if not response.is_successful(): print( "Query failed with status code {}".format(response.status_code)

- " with response {}".format(response.get_json()) ) raise Exception("Query failed") if len(result.hits) != 3: raise Exception("Expected 3 hits, got {}".format(len(result.hits))) print(json.dumps(result.hits, indent=4))

```
[
    {
        "id": "index:neighbors_content/0/c81e728dfde15fa4e8dfb3d3",
        "relevance": 5.385164807134504,
        "source": "neighbors_content",
        "fields": {
            "point": [
                2.0,
                3.0,
                4.0
            ]
        }
    },
    {
        "id": "index:neighbors_content/0/c4ca4238db266f395150e961",
        "relevance": 3.7416573867739413,
        "source": "neighbors_content",
        "fields": {
            "point": [
                1.0,
                2.0,
                3.0
            ]
        }
    },
    {
        "id": "index:neighbors_content/0/cfcd20845b10b1420c6cdeca",
        "relevance": 2.23606797749979,
        "source": "neighbors_content",
        "fields": {
            "point": [
                0.0,
                1.0,
                2.0
            ]
        }
    }
]
```

Query from `[1.0, 2.0, 2.9]` - find that `[2.0, 3.0, 4.0]` is most distant:

In \[17\]:

Copied!

```
result = app.query(
    body={
        "yql": "select point from neighbors where true",
        "input.query(qpoint)": "[1.0, 2.0, 2.9]",
        "ranking.profile": "max_distance",
        "presentation.format.tensors": "short-value",
    }
)
print(json.dumps(result.hits, indent=4))
```

result = app.query( body={ "yql": "select point from neighbors where true", "input.query(qpoint)": "[1.0, 2.0, 2.9]", "ranking.profile": "max_distance", "presentation.format.tensors": "short-value", } ) print(json.dumps(result.hits, indent=4))

```
[
    {
        "id": "index:neighbors_content/0/c81e728dfde15fa4e8dfb3d3",
        "relevance": 1.7916472308265357,
        "source": "neighbors_content",
        "fields": {
            "point": [
                2.0,
                3.0,
                4.0
            ]
        }
    },
    {
        "id": "index:neighbors_content/0/cfcd20845b10b1420c6cdeca",
        "relevance": 1.6763055154708881,
        "source": "neighbors_content",
        "fields": {
            "point": [
                0.0,
                1.0,
                2.0
            ]
        }
    },
    {
        "id": "index:neighbors_content/0/c4ca4238db266f395150e961",
        "relevance": 0.09999990575011103,
        "source": "neighbors_content",
        "fields": {
            "point": [
                1.0,
                2.0,
                3.0
            ]
        }
    }
]
```

### Nearest neighbor[¶](#nearest-neighbor)

The [nearestNeighbor](https://docs.vespa.ai/en/reference/query-language-reference.html#nearestneighbor) query operator calculates distances between points in vector space. Here, we are using the default distance metric (euclidean), as it is not specified. The [closeness()](<https://docs.vespa.ai/en/reference/rank-features.html#closeness(dimension,name)>) rank feature can be used to rank results - add a new rank profile:

In \[18\]:

Copied!

```
app_package.schema.add_rank_profile(
    RankProfile(
        name="nearest_neighbor",
        inputs=[("query(qpoint)", "tensor<float>(d[3])")],
        first_phase="closeness(field, point)",
    )
)

app = vespa_docker.deploy(application_package=app_package)
```

app_package.schema.add_rank_profile( RankProfile( name="nearest_neighbor", inputs=\[("query(qpoint)", "tensor<float>(d[3])")\], first_phase="closeness(field, point)", ) ) app = vespa_docker.deploy(application_package=app_package)

```
Waiting for configuration server, 0/300 seconds...
Waiting for configuration server, 5/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 0/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 5/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Application is up!
Finished deployment.
```

Read more in [nearest neighbor search](https://docs.vespa.ai/en/nearest-neighbor-search.html).

Query using nearestNeighbor query operator:

In \[19\]:

Copied!

```
result = app.query(
    body={
        "yql": "select point from neighbors where {targetHits: 3}nearestNeighbor(point, qpoint)",
        "input.query(qpoint)": "[1.0, 2.0, 2.9]",
        "ranking.profile": "nearest_neighbor",
        "presentation.format.tensors": "short-value",
    }
)
print(json.dumps(result.hits, indent=4))
```

result = app.query( body={ "yql": "select point from neighbors where {targetHits: 3}nearestNeighbor(point, qpoint)", "input.query(qpoint)": "[1.0, 2.0, 2.9]", "ranking.profile": "nearest_neighbor", "presentation.format.tensors": "short-value", } ) print(json.dumps(result.hits, indent=4))

```
[
    {
        "id": "index:neighbors_content/0/c4ca4238db266f395150e961",
        "relevance": 0.9090909879069752,
        "source": "neighbors_content",
        "fields": {
            "point": [
                1.0,
                2.0,
                3.0
            ]
        }
    },
    {
        "id": "index:neighbors_content/0/cfcd20845b10b1420c6cdeca",
        "relevance": 0.37364941905256455,
        "source": "neighbors_content",
        "fields": {
            "point": [
                0.0,
                1.0,
                2.0
            ]
        }
    },
    {
        "id": "index:neighbors_content/0/c81e728dfde15fa4e8dfb3d3",
        "relevance": 0.35821144946644456,
        "source": "neighbors_content",
        "fields": {
            "point": [
                2.0,
                3.0,
                4.0
            ]
        }
    }
]
```

### Nearest neighbor - angular[¶](#nearest-neighbor-angular)

So far, we have used the default [distance-metric](https://docs.vespa.ai/en/nearest-neighbor-search.html#distance-metrics-for-nearest-neighbor-search) which is euclidean - now try with another. Add new few field with "angular" distance metric:

In \[20\]:

Copied!

```
app_package.schema.add_fields(
    Field(
        name="point_angular",
        type="tensor<float>(d[3])",
        indexing=["attribute", "summary"],
        attribute=["distance-metric: angular"],
    )
)
app_package.schema.add_rank_profile(
    RankProfile(
        name="nearest_neighbor_angular",
        inputs=[("query(qpoint)", "tensor<float>(d[3])")],
        first_phase="closeness(field, point_angular)",
    )
)

app = vespa_docker.deploy(application_package=app_package)
```

app_package.schema.add_fields( Field( name="point_angular", type="tensor<float>(d[3])", indexing=["attribute", "summary"], attribute=["distance-metric: angular"], ) ) app_package.schema.add_rank_profile( RankProfile( name="nearest_neighbor_angular", inputs=\[("query(qpoint)", "tensor<float>(d[3])")\], first_phase="closeness(field, point_angular)", ) ) app = vespa_docker.deploy(application_package=app_package)

```
Waiting for configuration server, 0/300 seconds...
Waiting for configuration server, 5/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 0/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 5/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Waiting for application status, 10/300 seconds...
Using plain http against endpoint http://localhost:8080/ApplicationStatus
Application is up!
Finished deployment.
```

Feed the same data to the `point_angular` field:

In \[21\]:

Copied!

```
for u in get_feed("point_angular"):
    response: VespaResponse = session.update_data(
        data_id=u["id"], schema="neighbors", fields=u["fields"]
    )
    if not response.is_successful():
        print(
            "Update failed for document {}".format(u["id"])
            + " with status code {}".format(response.status_code)
            + " with response {}".format(response.get_json())
        )
```

for u in get_feed("point_angular"): response: VespaResponse = session.update_data( data_id=u["id"], schema="neighbors", fields=u["fields"] ) if not response.is_successful(): print( "Update failed for document {}".format(u["id"])

- " with status code {}".format(response.status_code)
- " with response {}".format(response.get_json()) )

Observe the documents now have *two* vectors

Notice that we pass [native Vespa document v1 api parameters](https://docs.vespa.ai/en/reference/document-v1-api-reference.html) to reduce the tensor verbosity.

In \[24\]:

Copied!

```
from vespa.io import VespaResponse

response: VespaResponse = app.get_data(
    schema="neighbors", data_id=0, **{"format.tensors": "short-value"}
)
print(json.dumps(response.get_json(), indent=4))
```

from vespa.io import VespaResponse response: VespaResponse = app.get_data( schema="neighbors", data_id=0, \*\*{"format.tensors": "short-value"} ) print(json.dumps(response.get_json(), indent=4))

```
{
    "pathId": "/document/v1/neighbors/neighbors/docid/0",
    "id": "id:neighbors:neighbors::0",
    "fields": {
        "point": [
            0.0,
            1.0,
            2.0
        ],
        "point_angular": [
            0.0,
            1.0,
            2.0
        ]
    }
}
```

In \[25\]:

Copied!

```
result = app.query(
    body={
        "yql": "select point_angular from neighbors where {targetHits: 3}nearestNeighbor(point_angular, qpoint)",
        "input.query(qpoint)": "[1.0, 2.0, 2.9]",
        "ranking.profile": "nearest_neighbor_angular",
        "presentation.format.tensors": "short-value",
    }
)
print(json.dumps(result.hits, indent=4))
```

result = app.query( body={ "yql": "select point_angular from neighbors where {targetHits: 3}nearestNeighbor(point_angular, qpoint)", "input.query(qpoint)": "[1.0, 2.0, 2.9]", "ranking.profile": "nearest_neighbor_angular", "presentation.format.tensors": "short-value", } ) print(json.dumps(result.hits, indent=4))

```
[
    {
        "id": "index:neighbors_content/0/c4ca4238db266f395150e961",
        "relevance": 0.983943389010042,
        "source": "neighbors_content",
        "fields": {
            "point_angular": [
                1.0,
                2.0,
                3.0
            ]
        }
    },
    {
        "id": "index:neighbors_content/0/c81e728dfde15fa4e8dfb3d3",
        "relevance": 0.9004871017951954,
        "source": "neighbors_content",
        "fields": {
            "point_angular": [
                2.0,
                3.0,
                4.0
            ]
        }
    },
    {
        "id": "index:neighbors_content/0/cfcd20845b10b1420c6cdeca",
        "relevance": 0.7638041096953281,
        "source": "neighbors_content",
        "fields": {
            "point_angular": [
                0.0,
                1.0,
                2.0
            ]
        }
    }
]
```

In the output above, observe the different in "relevance", compared to the query using `'ranking.profile': 'nearest_neighbor'` above - this is the difference in `closeness()` using different distance metrics.

## Next steps[¶](#next-steps)

- Try the [multi-vector-indexing](https://vespa-engine.github.io/pyvespa/examples/multi-vector-indexing.md) notebook to explore using an HNSW-index for *approximate* nearest neighbor search.
- Explore using the [distance()](<https://docs.vespa.ai/en/reference/rank-features.html#distance(dimension,name)>) rank feature - this should give the same results as the ranking expressions using `euclidean-distance` above.
- `label` is useful when having more vector fields - read more about the [nearestNeighbor](https://docs.vespa.ai/en/reference/query-language-reference.html#nearestneighbor) query operator.

## Cleanup[¶](#cleanup)

In \[ \]:

Copied!

```
vespa_docker.container.stop()
vespa_docker.container.remove()
```

vespa_docker.container.stop() vespa_docker.container.remove()
