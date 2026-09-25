# Evaluating retrieval with Snowflake arctic embed[¶](#evaluating-retrieval-with-snowflake-arctic-embed)

This notebook will demonstrate how different rank profiles in Vespa can be set up and evaluated. For the rank profiles that use semantic search, we will use the small version of [Snowflake's arctic embed model series](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) for generating embeddings.

Feel free to experiment with different sizes based on your need and compute/latency constraints.

> The snowflake-arctic-embedding models achieve state-of-the-art performance on the MTEB/BEIR leaderboard for each of their size variants.

We will set up and compare the following rank profiles:

- **unranked**: No ranking at all, for a dummy baseline.
- **bm25**: The classic BM25 ranker.
- **semantic**: Using `closeness(query_embedding, document_embedding)` only.
- **hybrid**: Combining BM25 and semantic search - `closeness(query_embedding, document_embedding) + log10( bm25(doc) )`.
- **hybrid_filter**: Same as the previous, but with a filter to exclude hits based on some heuristics.

In \[1\]:

Copied!

```
from vespa.package import (
    HNSW,
    ApplicationPackage,
    Component,
    Field,
    Parameter,
    Function,
)

app_name = "snowflake"

app_package = ApplicationPackage(
    name=app_name,
    components=[
        Component(
            id="snow",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    "transformer-model",
                    {
                        "url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-s/resolve/main/onnx/model_int8.onnx"
                    },
                ),
                Parameter(
                    "tokenizer-model",
                    {
                        "url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-s/raw/main/tokenizer.json"
                    },
                ),
                Parameter(
                    "normalize",
                    {},
                    "true",
                ),
                Parameter(
                    "pooling-strategy",
                    {},
                    "cls",
                ),
            ],
        )
    ],
)
```

from vespa.package import ( HNSW, ApplicationPackage, Component, Field, Parameter, Function, ) app_name = "snowflake" app_package = ApplicationPackage( name=app_name, components=\[ Component( id="snow", type="hugging-face-embedder", parameters=[ Parameter( "transformer-model", { "url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-s/resolve/main/onnx/model_int8.onnx" }, ), Parameter( "tokenizer-model", { "url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-s/raw/main/tokenizer.json" }, ), Parameter( "normalize", {}, "true", ), Parameter( "pooling-strategy", {}, "cls", ), ], ) \], )

In \[2\]:

Copied!

```
app_package.schema.add_fields(
    Field(name="id", type="int", indexing=["attribute", "summary"]),
    Field(
        name="doc", type="string", indexing=["index", "summary"], index="enable-bm25"
    ),
    Field(
        name="doc_embeddings",
        type="tensor<float>(x[384])",
        indexing=["input doc", "embed", "index", "attribute"],
        ann=HNSW(distance_metric="prenormalized-angular"),
        is_document_field=False,
    ),
)
```

app_package.schema.add_fields( Field(name="id", type="int", indexing=["attribute", "summary"]), Field( name="doc", type="string", indexing=["index", "summary"], index="enable-bm25" ), Field( name="doc_embeddings", type="tensor<float>(x[384])", indexing=["input doc", "embed", "index", "attribute"], ann=HNSW(distance_metric="prenormalized-angular"), is_document_field=False, ), )

In \[3\]:

Copied!

```
from vespa.package import (
    DocumentSummary,
    FieldSet,
    FirstPhaseRanking,
    RankProfile,
    SecondPhaseRanking,
    Summary,
)

app_package.schema.add_rank_profile(
    RankProfile(
        name="semantic",
        inputs=[("query(q)", "tensor<float>(x[384])")],
        inherits="default",
        first_phase="closeness(field, doc_embeddings)",
        match_features=["closeness(field, doc_embeddings)"],
    )
)

app_package.schema.add_rank_profile(RankProfile(name="bm25", first_phase="bm25(doc)"))
```

from vespa.package import ( DocumentSummary, FieldSet, FirstPhaseRanking, RankProfile, SecondPhaseRanking, Summary, ) app_package.schema.add_rank_profile( RankProfile( name="semantic", inputs=\[("query(q)", "tensor<float>(x[384])")\], inherits="default", first_phase="closeness(field, doc_embeddings)", match_features=["closeness(field, doc_embeddings)"], ) ) app_package.schema.add_rank_profile(RankProfile(name="bm25", first_phase="bm25(doc)"))

In \[4\]:

Copied!

```
app_package.schema.add_rank_profile(
    RankProfile(
        name="hybrid",
        inherits="semantic",
        # Guard against no keword match -> bm25 = 0 -> log10(0) = undefined
        functions=[
            Function(
                name="log_guard", expression="if (bm25(doc) > 0, log10(bm25(doc)), 0)"
            )
        ],
        first_phase=FirstPhaseRanking(expression="closeness(field, doc_embeddings)"),
        # Notice that we use log10 here, as the bm25 values with the natural logarithm tends to dominate the closeness values for these documents.
        second_phase=SecondPhaseRanking(expression="firstPhase + log_guard"),
        match_features=[
            "firstPhase",
            "bm25(doc)",
        ],
    )
)
```

app_package.schema.add_rank_profile( RankProfile( name="hybrid", inherits="semantic",

# Guard against no keword match -> bm25 = 0 -> log10(0) = undefined

functions=[ Function( name="log_guard", expression="if (bm25(doc) > 0, log10(bm25(doc)), 0)" ) ], first_phase=FirstPhaseRanking(expression="closeness(field, doc_embeddings)"),

# Notice that we use log10 here, as the bm25 values with the natural logarithm tends to dominate the closeness values for these documents.

second_phase=SecondPhaseRanking(expression="firstPhase + log_guard"), match_features=[ "firstPhase", "bm25(doc)", ], ) )

In \[5\]:

Copied!

```
app_package.schema.add_field_set(FieldSet(name="default", fields=["doc"]))
```

app_package.schema.add_field_set(FieldSet(name="default", fields=["doc"]))

In \[6\]:

Copied!

```
app_package.schema.add_document_summary(
    DocumentSummary(
        name="minimal",
        summary_fields=[Summary("id", "int"), Summary("doc", "string")],
    )
)
```

app_package.schema.add_document_summary( DocumentSummary( name="minimal", summary_fields=[Summary("id", "int"), Summary("doc", "string")], ) )

Create some sample documents that will help us see where the different ranking strategies have their strengths and weaknesses.

> These sample documents were created with a little help from ChatGPT.

Looking through the documents, we can see that a ranking of the documents in the order they are presented seem quite reasonable.

In \[7\]:

Copied!

```
# Query that the user is searching for
query = "How does Vespa handle real-time indexing and search?"

documents = [
    "Vespa excels in real-time data indexing and its ability to search large datasets quickly.",
    "Instant data availability and maintaining query performance while simultaneously indexing are key features of the Vespa search engine.",
    "With our search solution, real-time updates are seamlessly integrated into the search index, enhancing responsiveness.",
    "While not as robust as Vespa, our vector database strives to meet your search needs, despite certain, shall we say, 'flexible' features.",
    "Search engines like ours utilize complex algorithms to handle immediate data querying and indexing.",
    "Modern search platforms emphasize quick data retrieval from continuously updated indexes.",
    "Discover the history and cultural impact of the classic Italian Vespa scooter brand.",
    "Tips for maintaining your Vespa to ensure optimal performance and longevity of your scooter.",
    "Review of different scooter brands including Vespa, highlighting how they handle features like speed, cost, and aesthetics, and how consumers search for the best options.",
    "Vespa scooter safety regulations and best practices for urban commuting.",
]
```

# Query that the user is searching for

query = "How does Vespa handle real-time indexing and search?" documents = [ "Vespa excels in real-time data indexing and its ability to search large datasets quickly.", "Instant data availability and maintaining query performance while simultaneously indexing are key features of the Vespa search engine.", "With our search solution, real-time updates are seamlessly integrated into the search index, enhancing responsiveness.", "While not as robust as Vespa, our vector database strives to meet your search needs, despite certain, shall we say, 'flexible' features.", "Search engines like ours utilize complex algorithms to handle immediate data querying and indexing.", "Modern search platforms emphasize quick data retrieval from continuously updated indexes.", "Discover the history and cultural impact of the classic Italian Vespa scooter brand.", "Tips for maintaining your Vespa to ensure optimal performance and longevity of your scooter.", "Review of different scooter brands including Vespa, highlighting how they handle features like speed, cost, and aesthetics, and how consumers search for the best options.", "Vespa scooter safety regulations and best practices for urban commuting.", ]

## Dumping the application package to files[¶](#dumping-the-application-package-to-files)

This is a good practice to inspect and understand the structure of the application package and schema files, generated by pyvespa.

In \[8\]:

Copied!

```
app_package.to_files("snowflake")
```

app_package.to_files("snowflake")

In \[9\]:

Copied!

```
from vespa.deployment import VespaDocker

vespa_docker = VespaDocker()
app = vespa_docker.deploy(app_package)
```

from vespa.deployment import VespaDocker vespa_docker = VespaDocker() app = vespa_docker.deploy(app_package)

```
Waiting for configuration server, 0/60 seconds...
Waiting for configuration server, 5/60 seconds...
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

In \[10\]:

Copied!

```
feed_docs = [
    {
        "id": str(i),
        "fields": {
            "doc": doc,
        },
    }
    for i, doc in enumerate(documents)
]
```

feed_docs = [ { "id": str(i), "fields": { "doc": doc, }, } for i, doc in enumerate(documents) ]

In \[11\]:

Copied!

```
from vespa.io import VespaResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
        )
```

from vespa.io import VespaResponse def callback(response: VespaResponse, id: str): if not response.is_successful(): print( f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}" )

In \[12\]:

Copied!

```
app.feed_iterable(feed_docs, schema=app_package.schema.name, callback=callback)
```

app.feed_iterable(feed_docs, schema=app_package.schema.name, callback=callback)

## Choosing a metric[¶](#choosing-a-metric)

Check out [this](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29) wikipedia-article for an overview of evaluation measures in information retrieval.

In our case, we have a query and ranked documents as the ground truth.

When evaluating a ranking against our ground truth ranking, the Normalized Discounted Cumulative Gain (NDCG) metric is a good choice.

The NDCG is a measure of ranking quality. It is calculated as the sum of the discounted gain of the relevant documents(DCG), divided by the ideal DCG. The ideal DCG is the DCG of the perfect ranking, where the documents are ordered by relevance.

> If you are already familiar with NDCG, feel free to skip this part. There is also an implementation in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html) that you can use.

The formula for NDCG is:

$$ NDCG = \\frac{DCG}{IDCG} $$

where:

$$ DCG = \\sum\_{i=1}^{n} \\frac{2^{rel_i} - 1}{\\log_2(i + 1)} $$

Let us create a function to calculate the NDCG for a given ranking.

In \[13\]:

Copied!

```
import math
from typing import List


def ndcg_at_k(rank_order: List[int], ideal_order: List[int], k: int) -> float:
    """
    Calculate the normalized Discounted Cumulative Gain (nDCG) at position k.

    Parameters:
        rank_order (List[int]): The list of document indices as ranked by the search system.
        ideal_order (List[int]): The list of document indices in the ideal order.
        k (int): The position up to which to calculate nDCG.

    Returns:
        float: The nDCG value at position k.
    """
    dcg = 0.0
    idcg = 0.0

    # Calculate DCG based on the ranked order up to k
    for i in range(min(k, len(rank_order))):
        rank_index = rank_order[i]
        # Find the rank index in the ideal order to assign relevance
        if rank_index in ideal_order:
            relevance = len(ideal_order) - ideal_order.index(rank_index)
        else:
            relevance = 0
        dcg += relevance / math.log2(i + 2)

    # Calculate IDCG based on the ideal order up to k
    for i in range(min(k, len(ideal_order))):
        relevance = len(ideal_order) - i
        idcg += relevance / math.log2(i + 2)

    # Handle the case where IDCG is zero to avoid division by zero
    if idcg == 0:
        return 0.0
    return dcg / idcg


# Example usage
rank_order = [5, 6, 1]  # Example ranked order indices
ideal_result_order = [0, 1, 2, 4, 5, 3, 6, 7, 8, 9]  # Example ideal order indices

# Calculate nDCG@3
result = ndcg_at_k(rank_order, ideal_result_order, 3)
print(f"nDCG@3: {result:.4f}")

assert ndcg_at_k([0, 1, 2], ideal_result_order, 3) == 1.0
```

import math from typing import List def ndcg_at_k(rank_order: List[int], ideal_order: List[int], k: int) -> float: """ Calculate the normalized Discounted Cumulative Gain (nDCG) at position k. Parameters: rank_order (List[int]): The list of document indices as ranked by the search system. ideal_order (List[int]): The list of document indices in the ideal order. k (int): The position up to which to calculate nDCG. Returns: float: The nDCG value at position k. """ dcg = 0.0 idcg = 0.0

# Calculate DCG based on the ranked order up to k

for i in range(min(k, len(rank_order))): rank_index = rank_order[i]

# Find the rank index in the ideal order to assign relevance

if rank_index in ideal_order: relevance = len(ideal_order) - ideal_order.index(rank_index) else: relevance = 0 dcg += relevance / math.log2(i + 2)

# Calculate IDCG based on the ideal order up to k

for i in range(min(k, len(ideal_order))): relevance = len(ideal_order) - i idcg += relevance / math.log2(i + 2)

# Handle the case where IDCG is zero to avoid division by zero

if idcg == 0: return 0.0 return dcg / idcg

# Example usage

rank_order = [5, 6, 1] # Example ranked order indices ideal_result_order = [0, 1, 2, 4, 5, 3, 6, 7, 8, 9] # Example ideal order indices

# Calculate nDCG@3

result = ndcg_at_k(rank_order, ideal_result_order, 3) print(f"nDCG@3: {result:.4f}") assert ndcg_at_k([0, 1, 2], ideal_result_order, 3) == 1.0

```
nDCG@3: 0.6618
```

In \[14\]:

Copied!

```
# Define the different rank profiles to evaluate

rank_profiles = {
    "unranked": {
        "yql": f"select * from {app_name} where true",
        "ranking.profile": "unranked",
    },
    "bm25": {
        "yql": f"select * from {app_name} where userQuery()",
        "ranking.profile": "bm25",
    },
    "semantic": {
        "yql": f"select * from {app_name} where {{targetHits:5}}nearestNeighbor(doc_embeddings,q)",
        "ranking.profile": "semantic",
        "input.query(q)": f"embed({query})",
    },
    "hybrid": {
        "yql": f"select * from {app_name} where userQuery() or ({{targetHits:5}}nearestNeighbor(doc_embeddings,q))",
        "ranking.profile": "hybrid",
        "input.query(q)": f"embed({query})",
    },
    "hybrid_filtered": {
        # Here, we will add an heuristic, filtering out documents that contain the word "scooter"
        "yql": f'select * from {app_name} where !(doc contains "scooter") and userQuery() or ({{targetHits:5}}nearestNeighbor(doc_embeddings,q))',
        "ranking.profile": "hybrid",
        "input.query(q)": f"embed({query})",
    },
}

# Define some common params that will be used for all queries

common_params = {
    "query": query,
    "hits": 3,
}
```

# Define the different rank profiles to evaluate

rank_profiles = { "unranked": { "yql": f"select * from {app_name} where true", "ranking.profile": "unranked", }, "bm25": { "yql": f"select * from {app_name} where userQuery()", "ranking.profile": "bm25", }, "semantic": { "yql": f"select * from {app_name} where {{targetHits:5}}nearestNeighbor(doc_embeddings,q)", "ranking.profile": "semantic", "input.query(q)": f"embed({query})", }, "hybrid": { "yql": f"select * from {app_name} where userQuery() or ({{targetHits:5}}nearestNeighbor(doc_embeddings,q))", "ranking.profile": "hybrid", "input.query(q)": f"embed({query})", }, "hybrid_filtered": {

# Here, we will add an heuristic, filtering out documents that contain the word "scooter"

"yql": f'select * from {app_name} where !(doc contains "scooter") and userQuery() or ({{targetHits:5}}nearestNeighbor(doc_embeddings,q))', "ranking.profile": "hybrid", "input.query(q)": f"embed({query})", }, }

# Define some common params that will be used for all queries

common_params = { "query": query, "hits": 3, }

In \[15\]:

Copied!

```
from typing import List, Tuple

from vespa.application import Vespa
from vespa.io import VespaQueryResponse


def evaluate_rank_profile(
    app: Vespa, rank_profile: str, params: dict, k: int
) -> Tuple[float, List[str]]:
    """
    Run a query against a Vespa application using a specific rank profile and parameters.
    Evaluate the nDCG@3 of the search results based on the ideal order.

    Parameters:
        app (Vespa): The Vespa application to query.
        rank_profile (str): The name of the rank profile to use.
        params (dict): The common parameters to use in addition to the rank profile specific parameters.
        k (int): The position up to which to calculate nDCG.

    Returns:
        List[str]: The search results
    """
    body_params = {
        **rank_profiles[rank_profile],
        **params,
    }
    response: VespaQueryResponse = app.query(body_params)
    rankings = [int(hit["id"][-1]) for hit in response.hits]
    docs = [hit["fields"]["doc"] for hit in response.hits]
    ndcg = ndcg_at_k(rankings, ideal_order=ideal_result_order, k=3)
    return ndcg, docs
```

from typing import List, Tuple from vespa.application import Vespa from vespa.io import VespaQueryResponse def evaluate_rank_profile( app: Vespa, rank_profile: str, params: dict, k: int ) -> Tuple\[float, List[str]\]: """ Run a query against a Vespa application using a specific rank profile and parameters. Evaluate the nDCG@3 of the search results based on the ideal order. Parameters: app (Vespa): The Vespa application to query. rank_profile (str): The name of the rank profile to use. params (dict): The common parameters to use in addition to the rank profile specific parameters. k (int): The position up to which to calculate nDCG. Returns: List\[str\]: The search results """ body_params = { \*\*rank_profiles[rank_profile], \*\*params, } response: VespaQueryResponse = app.query(body_params) rankings = \[int(hit["id"][-1]) for hit in response.hits\] docs = \[hit["fields"]["doc"] for hit in response.hits\] ndcg = ndcg_at_k(rankings, ideal_order=ideal_result_order, k=3) return ndcg, docs

In \[16\]:

Copied!

```
import json

rank_results = {}

for rank_profile, params in rank_profiles.items():
    ndcg, docs = evaluate_rank_profile(
        app, rank_profile=rank_profile, params=common_params, k=3
    )
    rank_results[rank_profile] = ndcg
    print(f"Rank profile: {rank_profile}, nDCG@3: {ndcg:.2f}")
    print(json.dumps(docs, indent=2))
```

import json rank_results = {} for rank_profile, params in rank_profiles.items(): ndcg, docs = evaluate_rank_profile( app, rank_profile=rank_profile, params=common_params, k=3 ) rank_results[rank_profile] = ndcg print(f"Rank profile: {rank_profile}, nDCG@3: {ndcg:.2f}") print(json.dumps(docs, indent=2))

```
Rank profile: unranked, nDCG@3: 0.68
[
  "With our search solution, real-time updates are seamlessly integrated into the search index, enhancing responsiveness.",
  "Tips for maintaining your Vespa to ensure optimal performance and longevity of your scooter.",
  "Search engines like ours utilize complex algorithms to handle immediate data querying and indexing."
]
Rank profile: bm25, nDCG@3: 0.78
[
  "Vespa excels in real-time data indexing and its ability to search large datasets quickly.",
  "Review of different scooter brands including Vespa, highlighting how they handle features like speed, cost, and aesthetics, and how consumers search for the best options.",
  "With our search solution, real-time updates are seamlessly integrated into the search index, enhancing responsiveness."
]
Rank profile: semantic, nDCG@3: 0.94
[
  "Vespa excels in real-time data indexing and its ability to search large datasets quickly.",
  "With our search solution, real-time updates are seamlessly integrated into the search index, enhancing responsiveness.",
  "Search engines like ours utilize complex algorithms to handle immediate data querying and indexing."
]
Rank profile: hybrid, nDCG@3: 0.82
[
  "Vespa excels in real-time data indexing and its ability to search large datasets quickly.",
  "With our search solution, real-time updates are seamlessly integrated into the search index, enhancing responsiveness.",
  "Review of different scooter brands including Vespa, highlighting how they handle features like speed, cost, and aesthetics, and how consumers search for the best options."
]
Rank profile: hybrid_filtered, nDCG@3: 0.94
[
  "Vespa excels in real-time data indexing and its ability to search large datasets quickly.",
  "With our search solution, real-time updates are seamlessly integrated into the search index, enhancing responsiveness.",
  "Search engines like ours utilize complex algorithms to handle immediate data querying and indexing."
]
```

Uncomment the cell below to install dependencies needed to generate the plot.

In \[17\]:

Copied!

```
#!pip3 install pandas plotly
```

#!pip3 install pandas plotly

In \[20\]:

Copied!

```
import pandas as pd
import plotly.express as px


def plot_rank_profiles(rank_profiles):
    # Convert dictionary to DataFrame for easier manipulation
    data = pd.DataFrame(list(rank_profiles.items()), columns=["Rank Profile", "nDCG@3"])

    colors = {
        "unranked": "#e74c3c",  # Red
        "bm25": "#2ecc71",  # Green
        "semantic": "#9b59b6",  # Purple
        "hybrid": "#3498db",  # Blue
        "hybrid_filtered": "#2980b9",  # Darker Blue
    }

    # Map the colors to the dataframe
    data["Color"] = data["Rank Profile"].map(colors)

    # Create a bar chart using Plotly
    fig = px.bar(
        data,
        x="Rank Profile",
        y="nDCG@3",
        title="Rank Profile Performance - snowflake-arctic-embed-s",
        labels={"nDCG@3": "nDCG@3 Score"},
        text="nDCG@3",
        template="simple_white",
        color="Color",
        color_discrete_map="identity",
    )

    # Set bar width and update traces for individual colors
    fig.update_traces(
        marker_line_color="black", marker_line_width=1.5, width=0.4
    )  # Less wide bars

    # Enhance chart design adhering to Tufte's principles
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(
            title="Rank Profile",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
        ),
        yaxis=dict(
            title="nDCG@3 Score",
            range=[0, 1.1],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
        ),
        title_font=dict(size=24),
        font=dict(family="Arial, sans-serif", size=18, color="black"),
        margin=dict(l=40, r=40, t=40, b=40),
        width=800,  # Set the width of the plot
    )

    # Show the plot
    fig.show()


plot_rank_profiles(rank_profiles=rank_results)
```

import pandas as pd import plotly.express as px def plot_rank_profiles(rank_profiles):

# Convert dictionary to DataFrame for easier manipulation

data = pd.DataFrame(list(rank_profiles.items()), columns=["Rank Profile", "nDCG@3"]) colors = { "unranked": "#e74c3c", # Red "bm25": "#2ecc71", # Green "semantic": "#9b59b6", # Purple "hybrid": "#3498db", # Blue "hybrid_filtered": "#2980b9", # Darker Blue }

# Map the colors to the dataframe

data["Color"] = data["Rank Profile"].map(colors)

# Create a bar chart using Plotly

fig = px.bar( data, x="Rank Profile", y="nDCG@3", title="Rank Profile Performance - snowflake-arctic-embed-s", labels={"nDCG@3": "nDCG@3 Score"}, text="nDCG@3", template="simple_white", color="Color", color_discrete_map="identity", )

# Set bar width and update traces for individual colors

fig.update_traces( marker_line_color="black", marker_line_width=1.5, width=0.4 ) # Less wide bars

# Enhance chart design adhering to Tufte's principles

fig.update_traces(texttemplate="%{text:.2f}", textposition="outside") fig.update_layout( plot_bgcolor="white", xaxis=dict( title="Rank Profile", showline=True, linewidth=2, linecolor="black", mirror=True, ), yaxis=dict( title="nDCG@3 Score", range=[0, 1.1], showline=True, linewidth=2, linecolor="black", mirror=True, ), title_font=dict(size=24), font=dict(family="Arial, sans-serif", size=18, color="black"), margin=dict(l=40, r=40, t=40, b=40), width=800, # Set the width of the plot )

# Show the plot

fig.show() plot_rank_profiles(rank_profiles=rank_results)

For this particular synthetic small dataset, we can see that using the `snowflake-arctic-embed`-model improved the results significantly compared to keyword search only. Still, our experience with real-world data is that hybrid search is often the way to go.

We also provided a little taste of how one can evaluate different ranking profiles if you have a ground truth dataset available, (or can create a synthetic one).

## Next steps[¶](#next-steps)

Check out global reranking strategies, and try to introduce a global_phase reranking strategy.

## Cleanup[¶](#cleanup)

In \[19\]:

Copied!

```
vespa_docker.container.stop()
vespa_docker.container.remove()
```

vespa_docker.container.stop() vespa_docker.container.remove()
