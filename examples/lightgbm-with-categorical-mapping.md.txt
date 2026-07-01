# LightGBM: Mapping model features to Vespa features[¶](#lightgbm-mapping-model-features-to-vespa-features)

The main goal of this tutorial is to show how to deploy a LightGBM model with feature names that do not match Vespa feature names.

The following tasks will be accomplished throughout the tutorial:

1. Train a LightGBM classification model with generic feature names that will not be available in the Vespa application.
1. Create an application package and include a mapping from Vespa feature names to LightGBM model feature names.
1. Create Vespa application package files and export then to an application folder.
1. Export the trained LightGBM model to the Vespa application folder.
1. Deploy the Vespa application using the application folder.
1. Feed data to the Vespa application.
1. Assert that the LightGBM predictions from the deployed model are correct.

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

## Setup[¶](#setup)

Install and load required packages.

In \[ \]:

Copied!

```
!pip3 install numpy pandas pyvespa lightgbm
```

!pip3 install numpy pandas pyvespa lightgbm

In \[3\]:

Copied!

```
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
```

import json import lightgbm as lgb import numpy as np import pandas as pd

## Create data[¶](#create-data)

Simulate data that will be used to train the LightGBM model. Note that Vespa does not automatically recognize the feature names `feature_1`, `feature_2` and `feature_3`. When creating the application package we need to map those variables to something that the Vespa application recognizes, such as a document attribute or query value.

In \[4\]:

Copied!

```
# Create random training set
features = pd.DataFrame(
    {
        "feature_1": np.random.random(100),
        "feature_2": np.random.random(100),
        "feature_3": pd.Series(
            np.random.choice(["a", "b", "c"], size=100), dtype="category"
        ),
    }
)
features.head()
```

# Create random training set

features = pd.DataFrame( { "feature_1": np.random.random(100), "feature_2": np.random.random(100), "feature_3": pd.Series( np.random.choice(["a", "b", "c"], size=100), dtype="category" ), } ) features.head()

Out\[4\]:

|     | feature_1 | feature_2 | feature_3 |
| --- | --------- | --------- | --------- |
| 0   | 0.856415  | 0.550705  | a         |
| 1   | 0.615107  | 0.509030  | a         |
| 2   | 0.089759  | 0.667729  | c         |
| 3   | 0.161664  | 0.361693  | b         |
| 4   | 0.841505  | 0.967227  | b         |

Create a target variable that depends on `feature_1`, `feature_2` and `feature_3`:

In \[5\]:

Copied!

```
numeric_features = pd.get_dummies(features)
targets = (
    (
        numeric_features["feature_1"]
        + numeric_features["feature_2"]
        - 0.5 * numeric_features["feature_3_a"]
        + 0.5 * numeric_features["feature_3_c"]
    )
    > 1.0
) * 1.0
targets
```

numeric_features = pd.get_dummies(features) targets = ( ( numeric_features["feature_1"]

- numeric_features["feature_2"]

* 0.5 * numeric_features["feature_3_a"]

- 0.5 * numeric_features["feature_3_c"] )

> 1.0 ) * 1.0 targets

Out\[5\]:

```
0     0.0
1     0.0
2     1.0
3     0.0
4     1.0
     ... 
95    1.0
96    1.0
97    0.0
98    1.0
99    1.0
Length: 100, dtype: float64
```

## Fit lightgbm model[¶](#fit-lightgbm-model)

Train the LightGBM model on the simulated data,

In \[6\]:

Copied!

```
training_set = lgb.Dataset(features, targets)

# Train the model
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 3,
}
model = lgb.train(params, training_set, num_boost_round=5)
```

training_set = lgb.Dataset(features, targets)

# Train the model

params = { "objective": "binary", "metric": "binary_logloss", "num_leaves": 3, } model = lgb.train(params, training_set, num_boost_round=5)

```
[LightGBM] [Info] Number of positive: 48, number of negative: 52
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000404 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 74
[LightGBM] [Info] Number of data points in the train set: 100, number of used features: 3
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.480000 -> initscore=-0.080043
[LightGBM] [Info] Start training from score -0.080043
```

## Vespa application package[¶](#vespa-application-package)

Create the application package and map the LightGBM feature names to the related Vespa names.

In this example we are going to assume that `feature_1` represents the document field `numeric` and map `feature_1` to `attribute(numeric)` through the use of a Vespa `Function` in the corresponding `RankProfile`. `feature_2` maps to a `value` that will be sent along with the query, and this is represented in Vespa by mapping `query(value)` to `feature_2`. Lastly, the categorical feature is mapped from `attribute(categorical)` to `feature_3`.

In \[7\]:

Copied!

```
from vespa.package import ApplicationPackage, Field, RankProfile, Function

app_package = ApplicationPackage(name="lightgbm")
app_package.schema.add_fields(
    Field(name="id", type="string", indexing=["summary", "attribute"]),
    Field(name="numeric", type="double", indexing=["summary", "attribute"]),
    Field(name="categorical", type="string", indexing=["summary", "attribute"]),
)
app_package.schema.add_rank_profile(
    RankProfile(
        name="classify",
        functions=[
            Function(name="feature_1", expression="attribute(numeric)"),
            Function(name="feature_2", expression="query(value)"),
            Function(name="feature_3", expression="attribute(categorical)"),
        ],
        first_phase="lightgbm('lightgbm_model.json')",
    )
)
```

from vespa.package import ApplicationPackage, Field, RankProfile, Function app_package = ApplicationPackage(name="lightgbm") app_package.schema.add_fields( Field(name="id", type="string", indexing=["summary", "attribute"]), Field(name="numeric", type="double", indexing=["summary", "attribute"]), Field(name="categorical", type="string", indexing=["summary", "attribute"]), ) app_package.schema.add_rank_profile( RankProfile( name="classify", functions=[ Function(name="feature_1", expression="attribute(numeric)"), Function(name="feature_2", expression="query(value)"), Function(name="feature_3", expression="attribute(categorical)"), ], first_phase="lightgbm('lightgbm_model.json')", ) )

We can check how the Vespa search defition file will look like. Note that `feature_1`, `feature_2` and `feature_3` required by the LightGBM model are now defined on the schema definition:

In \[8\]:

Copied!

```
print(app_package.schema.schema_to_text)
```

print(app_package.schema.schema_to_text)

```
schema lightgbm {
    document lightgbm {
        field id type string {
            indexing: summary | attribute
        }
        field numeric type double {
            indexing: summary | attribute
        }
        field categorical type string {
            indexing: summary | attribute
        }
    }
    rank-profile classify {
        function feature_1() {
            expression {
                attribute(numeric)
            }
        }
        function feature_2() {
            expression {
                query(value)
            }
        }
        function feature_3() {
            expression {
                attribute(categorical)
            }
        }
        first-phase {
            expression {
                lightgbm('lightgbm_model.json')
            }
        }
    }
}
```

We can export the application package files to disk:

In \[9\]:

Copied!

```
from pathlib import Path

Path("lightgbm").mkdir(parents=True, exist_ok=True)
app_package.to_files("lightgbm")
```

from pathlib import Path Path("lightgbm").mkdir(parents=True, exist_ok=True) app_package.to_files("lightgbm")

Note that we don't have any models under the `models` folder. We need to export the lightGBM model that we trained earlier to `models/lightgbm.json`.

In \[13\]:

Copied!

```
!tree lightgbm
```

!tree lightgbm

```
lightgbm
├── files
├── models
│   └── lightgbm_model.json
├── schemas
│   └── lightgbm.sd
├── search
│   └── query-profiles
│       ├── default.xml
│       └── types
│           └── root.xml
└── services.xml

7 directories, 5 files
```

## Export the model[¶](#export-the-model)

In \[12\]:

Copied!

```
with open("lightgbm/models/lightgbm_model.json", "w") as f:
    json.dump(model.dump_model(), f, indent=2)
```

with open("lightgbm/models/lightgbm_model.json", "w") as f: json.dump(model.dump_model(), f, indent=2)

Now we can see that the model is where Vespa expects it to be:

In \[14\]:

Copied!

```
!tree lightgbm
```

!tree lightgbm

```
lightgbm
├── files
├── models
│   └── lightgbm_model.json
├── schemas
│   └── lightgbm.sd
├── search
│   └── query-profiles
│       ├── default.xml
│       └── types
│           └── root.xml
└── services.xml

7 directories, 5 files
```

## Deploy the application[¶](#deploy-the-application)

Deploy the application package from disk with Docker:

In \[15\]:

Copied!

```
from vespa.deployment import VespaDocker

vespa_docker = VespaDocker()
app = vespa_docker.deploy_from_disk(
    application_name="lightgbm", application_root="lightgbm"
)
```

from vespa.deployment import VespaDocker vespa_docker = VespaDocker() app = vespa_docker.deploy_from_disk( application_name="lightgbm", application_root="lightgbm" )

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

## Feed the data[¶](#feed-the-data)

Feed the simulated data. To feed data in batch we need to create a list of dictionaries containing id and fields keys:

In \[16\]:

Copied!

```
feed_batch = [
    {
        "id": idx,
        "fields": {
            "id": idx,
            "numeric": row["feature_1"],
            "categorical": row["feature_3"],
        },
    }
    for idx, row in features.iterrows()
]
```

feed_batch = \[ { "id": idx, "fields": { "id": idx, "numeric": row["feature_1"], "categorical": row["feature_3"], }, } for idx, row in features.iterrows() \]

In \[17\]:

Copied!

```
from vespa.io import VespaResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Document {id} was not fed to Vespa due to error: {response.get_json()}")


app.feed_iterable(feed_batch, callback=callback)
```

from vespa.io import VespaResponse def callback(response: VespaResponse, id: str): if not response.is_successful(): print(f"Document {id} was not fed to Vespa due to error: {response.get_json()}") app.feed_iterable(feed_batch, callback=callback)

## Model predictions[¶](#model-predictions)

Predict with the trained LightGBM model so that we can later compare with the predictions returned by Vespa.

In \[18\]:

Copied!

```
features["model_prediction"] = model.predict(features)
```

features["model_prediction"] = model.predict(features)

In \[19\]:

Copied!

```
features
```

features

Out\[19\]:

|     | feature_1 | feature_2 | feature_3 | model_prediction |
| --- | --------- | --------- | --------- | ---------------- |
| 0   | 0.856415  | 0.550705  | a         | 0.402572         |
| 1   | 0.615107  | 0.509030  | a         | 0.356262         |
| 2   | 0.089759  | 0.667729  | c         | 0.641578         |
| 3   | 0.161664  | 0.361693  | b         | 0.388184         |
| 4   | 0.841505  | 0.967227  | b         | 0.632525         |
| ... | ...       | ...       | ...       | ...              |
| 95  | 0.087768  | 0.451850  | c         | 0.641578         |
| 96  | 0.839063  | 0.644387  | b         | 0.632525         |
| 97  | 0.725573  | 0.327668  | a         | 0.376350         |
| 98  | 0.937481  | 0.199995  | b         | 0.376350         |
| 99  | 0.918530  | 0.734004  | a         | 0.402572         |

100 rows × 4 columns

## Query[¶](#query)

Create a `compute_vespa_relevance` function that takes a document `id` and a query `value` and return the LightGBM model deployed.

In \[20\]:

Copied!

```
def compute_vespa_relevance(id_value: int):
    hits = app.query(
        body={
            "yql": "select * from sources * where id = {}".format(str(id_value)),
            "ranking": "classify",
            "ranking.features.query(value)": features.loc[id_value, "feature_2"],
            "hits": 1,
        }
    ).hits
    return hits[0]["relevance"]


compute_vespa_relevance(id_value=0)
```

def compute_vespa_relevance(id_value: int): hits = app.query( body={ "yql": "select * from sources * where id = {}".format(str(id_value)), "ranking": "classify", "ranking.features.query(value)": features.loc[id_value, "feature_2"], "hits": 1, } ).hits return hits[0]["relevance"] compute_vespa_relevance(id_value=0)

Out\[20\]:

```
0.4025720849980601
```

Loop through the `features` to compute a vespa prediction for all the data points, so that we can compare it to the predictions made by the model outside Vespa.

In \[21\]:

Copied!

```
vespa_relevance = []
for idx, row in features.iterrows():
    vespa_relevance.append(compute_vespa_relevance(id_value=idx))
features["vespa_relevance"] = vespa_relevance
```

vespa_relevance = [] for idx, row in features.iterrows(): vespa_relevance.append(compute_vespa_relevance(id_value=idx)) features["vespa_relevance"] = vespa_relevance

In \[22\]:

Copied!

```
features
```

features

Out\[22\]:

|     | feature_1 | feature_2 | feature_3 | model_prediction | vespa_relevance |
| --- | --------- | --------- | --------- | ---------------- | --------------- |
| 0   | 0.856415  | 0.550705  | a         | 0.402572         | 0.402572        |
| 1   | 0.615107  | 0.509030  | a         | 0.356262         | 0.356262        |
| 2   | 0.089759  | 0.667729  | c         | 0.641578         | 0.641578        |
| 3   | 0.161664  | 0.361693  | b         | 0.388184         | 0.388184        |
| 4   | 0.841505  | 0.967227  | b         | 0.632525         | 0.632525        |
| ... | ...       | ...       | ...       | ...              | ...             |
| 95  | 0.087768  | 0.451850  | c         | 0.641578         | 0.641578        |
| 96  | 0.839063  | 0.644387  | b         | 0.632525         | 0.632525        |
| 97  | 0.725573  | 0.327668  | a         | 0.376350         | 0.376350        |
| 98  | 0.937481  | 0.199995  | b         | 0.376350         | 0.376350        |
| 99  | 0.918530  | 0.734004  | a         | 0.402572         | 0.402572        |

100 rows × 5 columns

## Compare model and Vespa predictions[¶](#compare-model-and-vespa-predictions)

Predictions from the model should be equal to predictions from Vespa, showing the model was correctly deployed to Vespa.

In \[ \]:

Copied!

```
# Use numpy's allclose for floating-point comparison with tolerance
assert np.allclose(
    features["model_prediction"].values,
    features["vespa_relevance"].values,
    rtol=1e-9,
    atol=1e-15,
), "Model predictions and Vespa relevance values should be approximately equal"
```

# Use numpy's allclose for floating-point comparison with tolerance

assert np.allclose( features["model_prediction"].values, features["vespa_relevance"].values, rtol=1e-9, atol=1e-15, ), "Model predictions and Vespa relevance values should be approximately equal"

## Clean environment[¶](#clean-environment)

In \[24\]:

Copied!

```
!rm -fr lightgbm
vespa_docker.container.stop()
vespa_docker.container.remove()
```

!rm -fr lightgbm vespa_docker.container.stop() vespa_docker.container.remove()
