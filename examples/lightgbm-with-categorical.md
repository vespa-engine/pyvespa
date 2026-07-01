# LightGBM: Training the model with Vespa features[¶](#lightgbm-training-the-model-with-vespa-features)

The main goal of this tutorial is to deploy and use a LightGBM model in a Vespa application. The following tasks will be accomplished throughout the tutorial:

1. Train a LightGBM classification model with variable names supported by Vespa.
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

Generate a toy dataset to follow along. Note that we set the column names in a format that Vespa understands. `query(value)` means that the user will send a parameter named `value` along with the query. `attribute(field)` means that `field` is a document attribute defined in a schema. In the example below we have a query parameter named `value` and two document's attributes, `numeric` and `categorical`. If we want `lightgbm` to handle categorical variables we should use `dtype="category"` when creating the dataframe, as shown below.

In \[4\]:

Copied!

```
# Create random training set
features = pd.DataFrame(
    {
        "query(value)": np.random.random(100),
        "attribute(numeric)": np.random.random(100),
        "attribute(categorical)": pd.Series(
            np.random.choice(["a", "b", "c"], size=100), dtype="category"
        ),
    }
)
features.head()
```

# Create random training set

features = pd.DataFrame( { "query(value)": np.random.random(100), "attribute(numeric)": np.random.random(100), "attribute(categorical)": pd.Series( np.random.choice(["a", "b", "c"], size=100), dtype="category" ), } ) features.head()

Out\[4\]:

|     | query(value) | attribute(numeric) | attribute(categorical) |
| --- | ------------ | ------------------ | ---------------------- |
| 0   | 0.437748     | 0.442222           | c                      |
| 1   | 0.957135     | 0.323047           | b                      |
| 2   | 0.514168     | 0.426117           | a                      |
| 3   | 0.713511     | 0.886630           | b                      |
| 4   | 0.626918     | 0.663179           | c                      |

We generate the target variable as a function of the three features defined above:

In \[5\]:

Copied!

```
numeric_features = pd.get_dummies(features)
targets = (
    (
        numeric_features["query(value)"]
        + numeric_features["attribute(numeric)"]
        - 0.5 * numeric_features["attribute(categorical)_a"]
        + 0.5 * numeric_features["attribute(categorical)_c"]
    )
    > 1.0
) * 1.0
targets
```

numeric_features = pd.get_dummies(features) targets = ( ( numeric_features["query(value)"]

- numeric_features["attribute(numeric)"]

* 0.5 * numeric_features["attribute(categorical)\_a"]

- 0.5 * numeric_features["attribute(categorical)\_c"] )

> 1.0 ) * 1.0 targets

Out\[5\]:

```
0     1.0
1     1.0
2     0.0
3     1.0
4     1.0
     ... 
95    0.0
96    1.0
97    0.0
98    0.0
99    1.0
Length: 100, dtype: float64
```

## Fit lightgbm model[¶](#fit-lightgbm-model)

Train an LightGBM model with a binary loss function:

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
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000484 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 74
[LightGBM] [Info] Number of data points in the train set: 100, number of used features: 3
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.480000 -> initscore=-0.080043
[LightGBM] [Info] Start training from score -0.080043
```

## Vespa application package[¶](#vespa-application-package)

Create a Vespa application package. The model expects two document attributes, `numeric` and `categorical`. We can use the model in the first-phase ranking by using the `lightgbm` rank feature.

In \[7\]:

Copied!

```
from vespa.package import ApplicationPackage, Field, RankProfile

app_package = ApplicationPackage(name="lightgbm")
app_package.schema.add_fields(
    Field(name="id", type="string", indexing=["summary", "attribute"]),
    Field(name="numeric", type="double", indexing=["summary", "attribute"]),
    Field(name="categorical", type="string", indexing=["summary", "attribute"]),
)
app_package.schema.add_rank_profile(
    RankProfile(name="classify", first_phase="lightgbm('lightgbm_model.json')")
)
```

from vespa.package import ApplicationPackage, Field, RankProfile app_package = ApplicationPackage(name="lightgbm") app_package.schema.add_fields( Field(name="id", type="string", indexing=["summary", "attribute"]), Field(name="numeric", type="double", indexing=["summary", "attribute"]), Field(name="categorical", type="string", indexing=["summary", "attribute"]), ) app_package.schema.add_rank_profile( RankProfile(name="classify", first_phase="lightgbm('lightgbm_model.json')") )

We can check how the Vespa search defition file will look like:

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

In \[10\]:

Copied!

```
!tree lightgbm
```

!tree lightgbm

```
lightgbm
├── files
├── models
├── schemas
│   └── lightgbm.sd
├── search
│   └── query-profiles
│       ├── default.xml
│       └── types
│           └── root.xml
└── services.xml

7 directories, 4 files
```

## Export the model[¶](#export-the-model)

In \[11\]:

Copied!

```
with open("lightgbm/models/lightgbm_model.json", "w") as f:
    json.dump(model.dump_model(), f, indent=2)
```

with open("lightgbm/models/lightgbm_model.json", "w") as f: json.dump(model.dump_model(), f, indent=2)

Now we can see that the model is where Vespa expects it to be:

In \[12\]:

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

In \[13\]:

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

## Feed the data[¶](#feed-the-data)

Feed the simulated data. To feed data in batch we need to create a list of dictionaries containing `id` and `fields` keys:

In \[14\]:

Copied!

```
feed_batch = [
    {
        "id": idx,
        "fields": {
            "id": idx,
            "numeric": row["attribute(numeric)"],
            "categorical": row["attribute(categorical)"],
        },
    }
    for idx, row in features.iterrows()
]
```

feed_batch = \[ { "id": idx, "fields": { "id": idx, "numeric": row["attribute(numeric)"], "categorical": row["attribute(categorical)"], }, } for idx, row in features.iterrows() \]

Feed the batch of data:

In \[15\]:

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

In \[16\]:

Copied!

```
features["model_prediction"] = model.predict(features)
```

features["model_prediction"] = model.predict(features)

In \[17\]:

Copied!

```
features
```

features

Out\[17\]:

|     | query(value) | attribute(numeric) | attribute(categorical) | model_prediction |
| --- | ------------ | ------------------ | ---------------------- | ---------------- |
| 0   | 0.437748     | 0.442222           | c                      | 0.645663         |
| 1   | 0.957135     | 0.323047           | b                      | 0.645663         |
| 2   | 0.514168     | 0.426117           | a                      | 0.354024         |
| 3   | 0.713511     | 0.886630           | b                      | 0.645663         |
| 4   | 0.626918     | 0.663179           | c                      | 0.645663         |
| ... | ...          | ...                | ...                    | ...              |
| 95  | 0.208583     | 0.103319           | c                      | 0.352136         |
| 96  | 0.882902     | 0.224213           | c                      | 0.645663         |
| 97  | 0.604831     | 0.675583           | a                      | 0.354024         |
| 98  | 0.278674     | 0.008019           | b                      | 0.352136         |
| 99  | 0.417318     | 0.616241           | b                      | 0.645663         |

100 rows × 4 columns

## Query[¶](#query)

Create a `compute_vespa_relevance` function that takes a document `id` and a query `value` and return the LightGBM model deployed.

In \[18\]:

Copied!

```
def compute_vespa_relevance(id_value: int):
    hits = app.query(
        body={
            "yql": "select * from sources * where id = {}".format(str(id_value)),
            "ranking": "classify",
            "ranking.features.query(value)": features.loc[id_value, "query(value)"],
            "hits": 1,
        }
    ).hits
    return hits[0]["relevance"]


compute_vespa_relevance(id_value=0)
```

def compute_vespa_relevance(id_value: int): hits = app.query( body={ "yql": "select * from sources * where id = {}".format(str(id_value)), "ranking": "classify", "ranking.features.query(value)": features.loc[id_value, "query(value)"], "hits": 1, } ).hits return hits[0]["relevance"] compute_vespa_relevance(id_value=0)

Out\[18\]:

```
0.645662636917761
```

Loop through the `features` to compute a vespa prediction for all the data points, so that we can compare it to the predictions made by the model outside Vespa.

In \[19\]:

Copied!

```
vespa_relevance = []
for idx, row in features.iterrows():
    vespa_relevance.append(compute_vespa_relevance(id_value=idx))
features["vespa_relevance"] = vespa_relevance
```

vespa_relevance = [] for idx, row in features.iterrows(): vespa_relevance.append(compute_vespa_relevance(id_value=idx)) features["vespa_relevance"] = vespa_relevance

In \[20\]:

Copied!

```
features
```

features

Out\[20\]:

|     | query(value) | attribute(numeric) | attribute(categorical) | model_prediction | vespa_relevance |
| --- | ------------ | ------------------ | ---------------------- | ---------------- | --------------- |
| 0   | 0.437748     | 0.442222           | c                      | 0.645663         | 0.645663        |
| 1   | 0.957135     | 0.323047           | b                      | 0.645663         | 0.645663        |
| 2   | 0.514168     | 0.426117           | a                      | 0.354024         | 0.354024        |
| 3   | 0.713511     | 0.886630           | b                      | 0.645663         | 0.645663        |
| 4   | 0.626918     | 0.663179           | c                      | 0.645663         | 0.645663        |
| ... | ...          | ...                | ...                    | ...              | ...             |
| 95  | 0.208583     | 0.103319           | c                      | 0.352136         | 0.352136        |
| 96  | 0.882902     | 0.224213           | c                      | 0.645663         | 0.645663        |
| 97  | 0.604831     | 0.675583           | a                      | 0.354024         | 0.354024        |
| 98  | 0.278674     | 0.008019           | b                      | 0.352136         | 0.352136        |
| 99  | 0.417318     | 0.616241           | b                      | 0.645663         | 0.645663        |

100 rows × 5 columns

## Compare model and Vespa predictions[¶](#compare-model-and-vespa-predictions)

Predictions from the model should be equal to predictions from Vespa, showing the model was correctly deployed to Vespa.

In \[ \]:

Copied!

```
assert np.allclose(features["model_prediction"], features["vespa_relevance"])
```

assert np.allclose(features["model_prediction"], features["vespa_relevance"])

## Clean environment[¶](#clean-environment)

In \[22\]:

Copied!

```
!rm -fr lightgbm
vespa_docker.container.stop()
vespa_docker.container.remove()
```

!rm -fr lightgbm vespa_docker.container.stop() vespa_docker.container.remove()
