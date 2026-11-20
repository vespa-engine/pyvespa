# Multi-vector indexing with HNSW[¶](#multi-vector-indexing-with-hnsw)

This is the pyvespa steps of the multi-vector-indexing sample application. Go to the [source](https://github.com/vespa-engine/sample-apps/tree/master/multi-vector-indexing) for a full description and prerequisites, and read the [blog post](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/). Highlighted features:

- Approximate Nearest Neighbor Search - using HNSW or exact
- Use a Component to configure the Huggingface embedder.
- Using synthetic fields with auto-generated [embeddings](https://docs.vespa.ai/en/embedding.html) in data and query flow.
- Application package file export, model files in the application package, deployment from files.
- [Multiphased ranking](https://docs.vespa.ai/en/phased-ranking.html).
- How to control text search result highlighting.

For simpler examples, see [text search](https://vespa-engine.github.io/pyvespa/getting-started-pyvespa.md) and [pyvespa examples](https://vespa-engine.github.io/pyvespa/examples/pyvespa-examples.md).

Pyvespa is an add-on to Vespa, and this guide will export the application package containing `services.xml` and `wiki.sd`. The latter is the schema file for this application - knowing services.xml and schema files is useful when reading Vespa documentation.

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

This notebook requires [pyvespa >= 0.37.1](https://vespa-engine.github.io/pyvespa/index.md#requirements), ZSTD, and the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html).

In \[ \]:

Copied!

```
!pip3 install pyvespa
```

!pip3 install pyvespa

## Create the application[¶](#create-the-application)

Configure the Vespa instance with a component loading the E5-small model. Components are used to plug in code and models to a Vespa application - [read more](https://docs.vespa.ai/en/jdisc/container-components.html):

In \[1\]:

Copied!

```
from vespa.package import (
    ApplicationPackage,
    Component,
    Parameter,
    Field,
    HNSW,
    RankProfile,
    Function,
    FirstPhaseRanking,
    SecondPhaseRanking,
    FieldSet,
    DocumentSummary,
    Summary,
)
from pathlib import Path
import json

app_package = ApplicationPackage(
    name="wiki",
    components=[
        Component(
            id="e5-small-q",
            type="hugging-face-embedder",
            parameters=[
                Parameter("transformer-model", {"path": "model/e5-small-v2-int8.onnx"}),
                Parameter("tokenizer-model", {"path": "model/tokenizer.json"}),
            ],
        )
    ],
)
```

from vespa.package import ( ApplicationPackage, Component, Parameter, Field, HNSW, RankProfile, Function, FirstPhaseRanking, SecondPhaseRanking, FieldSet, DocumentSummary, Summary, ) from pathlib import Path import json app_package = ApplicationPackage( name="wiki", components=\[ Component( id="e5-small-q", type="hugging-face-embedder", parameters=[ Parameter("transformer-model", {"path": "model/e5-small-v2-int8.onnx"}), Parameter("tokenizer-model", {"path": "model/tokenizer.json"}), ], ) \], )

## Configure fields[¶](#configure-fields)

Vespa has a variety of basic and complex [field types](https://docs.vespa.ai/en/reference/schema-reference.html#field). This application uses a combination of integer, text and tensor fields, making it easy to implement hybrid ranking use cases:

In \[2\]:

Copied!

```
app_package.schema.add_fields(
    Field(name="id", type="int", indexing=["attribute", "summary"]),
    Field(
        name="title", type="string", indexing=["index", "summary"], index="enable-bm25"
    ),
    Field(
        name="url", type="string", indexing=["index", "summary"], index="enable-bm25"
    ),
    Field(
        name="paragraphs",
        type="array<string>",
        indexing=["index", "summary"],
        index="enable-bm25",
        bolding=True,
    ),
    Field(
        name="paragraph_embeddings",
        type="tensor<float>(p{},x[384])",
        indexing=["input paragraphs", "embed", "index", "attribute"],
        ann=HNSW(distance_metric="angular"),
        is_document_field=False,
    ),
    #
    # Alteratively, for exact distance calculation not using HNSW:
    #
    # Field(name="paragraph_embeddings", type="tensor<float>(p{},x[384])",
    #       indexing=["input paragraphs", "embed", "attribute"],
    #       attribute=["distance-metric: angular"],
    #       is_document_field=False)
)
```

app_package.schema.add_fields( Field(name="id", type="int", indexing=["attribute", "summary"]), Field( name="title", type="string", indexing=["index", "summary"], index="enable-bm25" ), Field( name="url", type="string", indexing=["index", "summary"], index="enable-bm25" ), Field( name="paragraphs", type="array<string>", indexing=["index", "summary"], index="enable-bm25", bolding=True, ), Field( name="paragraph_embeddings", type="tensor<float>(p{},x[384])", indexing=["input paragraphs", "embed", "index", "attribute"], ann=HNSW(distance_metric="angular"), is_document_field=False, ),

# 

# Alteratively, for exact distance calculation not using HNSW:

# 

# Field(name="paragraph_embeddings", type="tensor<float>(p{},x[384])",

# indexing=["input paragraphs", "embed", "attribute"],

# attribute=["distance-metric: angular"],

# is_document_field=False)

)

One field of particular interest is `paragraph_embeddings`. Note that we are *not* feeding embeddings to this instance. Instead, the embeddings are generated by using the [embed](https://docs.vespa.ai/en/embedding.html) feature, using the model configured at start. Read more in [Text embedding made simple](https://blog.vespa.ai/text-embedding-made-simple/).

Looking closely at the code, `paragraph_embeddings` uses `is_document_field=False`, meaning it will read another field as input (here `paragraph`), and run `embed` on it.

As only one model is configured, `embed` will use that one - it is possible to configure mode models and use `embed model-id` as well.

As the code comment illustrates, there can be different distrance metrics used, as well as using an *exact* or *approximate* nearest neighbor search.

## Configure rank profiles[¶](#configure-rank-profiles)

A rank profile defines the computation for the ranking, with a wide range of possible features as input. Below you will find `first_phase` ranking using text ranking (`bm`), semantic ranking using vector distance (consider a tensor a vector here), and combinations of the two:

In \[3\]:

Copied!

```
app_package.schema.add_rank_profile(
    RankProfile(
        name="semantic",
        inputs=[("query(q)", "tensor<float>(x[384])")],
        inherits="default",
        first_phase="cos(distance(field,paragraph_embeddings))",
        match_features=["closest(paragraph_embeddings)"],
    )
)

app_package.schema.add_rank_profile(
    RankProfile(name="bm25", first_phase="2*bm25(title) + bm25(paragraphs)")
)

app_package.schema.add_rank_profile(
    RankProfile(
        name="hybrid",
        inherits="semantic",
        functions=[
            Function(
                name="avg_paragraph_similarity",
                expression="""reduce(
                              sum(l2_normalize(query(q),x) * l2_normalize(attribute(paragraph_embeddings),x),x),
                              avg,
                              p
                          )""",
            ),
            Function(
                name="max_paragraph_similarity",
                expression="""reduce(
                              sum(l2_normalize(query(q),x) * l2_normalize(attribute(paragraph_embeddings),x),x),
                              max,
                              p
                          )""",
            ),
            Function(
                name="all_paragraph_similarities",
                expression="sum(l2_normalize(query(q),x) * l2_normalize(attribute(paragraph_embeddings),x),x)",
            ),
        ],
        first_phase=FirstPhaseRanking(
            expression="cos(distance(field,paragraph_embeddings))"
        ),
        second_phase=SecondPhaseRanking(
            expression="firstPhase + avg_paragraph_similarity() + log( bm25(title) + bm25(paragraphs) + bm25(url))"
        ),
        match_features=[
            "closest(paragraph_embeddings)",
            "firstPhase",
            "bm25(title)",
            "bm25(paragraphs)",
            "avg_paragraph_similarity",
            "max_paragraph_similarity",
            "all_paragraph_similarities",
        ],
    )
)
```

app_package.schema.add_rank_profile( RankProfile( name="semantic", inputs=\[("query(q)", "tensor<float>(x[384])")\], inherits="default", first_phase="cos(distance(field,paragraph_embeddings))", match_features=["closest(paragraph_embeddings)"], ) ) app_package.schema.add_rank_profile( RankProfile(name="bm25", first_phase="2\*bm25(title) + bm25(paragraphs)") ) app_package.schema.add_rank_profile( RankProfile( name="hybrid", inherits="semantic", functions=[ Function( name="avg_paragraph_similarity", expression="""reduce( sum(l2_normalize(query(q),x) * l2_normalize(attribute(paragraph_embeddings),x),x), avg, p )""", ), Function( name="max_paragraph_similarity", expression="""reduce( sum(l2_normalize(query(q),x) * l2_normalize(attribute(paragraph_embeddings),x),x), max, p )""", ), Function( name="all_paragraph_similarities", expression="sum(l2_normalize(query(q),x) * l2_normalize(attribute(paragraph_embeddings),x),x)", ), ], first_phase=FirstPhaseRanking( expression="cos(distance(field,paragraph_embeddings))" ), second_phase=SecondPhaseRanking( expression="firstPhase + avg_paragraph_similarity() + log( bm25(title) + bm25(paragraphs) + bm25(url))" ), match_features=[ "closest(paragraph_embeddings)", "firstPhase", "bm25(title)", "bm25(paragraphs)", "avg_paragraph_similarity", "max_paragraph_similarity", "all_paragraph_similarities", ], ) )

## Configure fieldset[¶](#configure-fieldset)

A [fieldset](https://docs.vespa.ai/en/reference/schema-reference.html#fieldset) is a way to configure search in multiple fields:

In \[4\]:

Copied!

```
app_package.schema.add_field_set(
    FieldSet(name="default", fields=["title", "url", "paragraphs"])
)
```

app_package.schema.add_field_set( FieldSet(name="default", fields=["title", "url", "paragraphs"]) )

## Configure document summary[¶](#configure-document-summary)

A [document summary](https://docs.vespa.ai/en/document-summaries.html) is the collection of fields to return in query results - the default summary is used unless other specified in the query. Here we configure a `minimal` fieldset without the larger paragraph text/embedding fields:

In \[5\]:

Copied!

```
app_package.schema.add_document_summary(
    DocumentSummary(
        name="minimal",
        summary_fields=[Summary("id", "int"), Summary("title", "string")],
    )
)
```

app_package.schema.add_document_summary( DocumentSummary( name="minimal", summary_fields=[Summary("id", "int"), Summary("title", "string")], ) )

## Export the configuration[¶](#export-the-configuration)

At this point, the application is well defined. Remember that the Component configuration at start configures model files to be found in a `model` directory. We must therefore export the configuration and add the models, before we can deploy to the Vespa instance. Export the [application package](https://docs.vespa.ai/en/application-packages.html):

In \[6\]:

Copied!

```
Path("pkg").mkdir(parents=True, exist_ok=True)
app_package.to_files("pkg")
```

Path("pkg").mkdir(parents=True, exist_ok=True) app_package.to_files("pkg")

It is a good idea to inspect the files exported into `pkg` - these are files referred to in the [Vespa Documentation](https://docs.vespa.ai/).

## Download model files[¶](#download-model-files)

At this point, we can save the model files into the application package:

In \[7\]:

Copied!

```
! mkdir -p pkg/model
! curl -L -o pkg/model/tokenizer.json \
  https://raw.githubusercontent.com/vespa-engine/sample-apps/master/examples/model-exporting/model/tokenizer.json

! curl -L -o pkg/model/e5-small-v2-int8.onnx \
  https://github.com/vespa-engine/sample-apps/raw/master/examples/model-exporting/model/e5-small-v2-int8.onnx
```

! mkdir -p pkg/model ! curl -L -o pkg/model/tokenizer.json \
https://raw.githubusercontent.com/vespa-engine/sample-apps/master/examples/model-exporting/model/tokenizer.json ! curl -L -o pkg/model/e5-small-v2-int8.onnx \
https://github.com/vespa-engine/sample-apps/raw/master/examples/model-exporting/model/e5-small-v2-int8.onnx

```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  694k  100  694k    0     0  2473k      0 --:--:-- --:--:-- --:--:-- 2508k
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100 32.3M  100 32.3M    0     0  27.1M      0  0:00:01  0:00:01 --:--:-- 53.0M
```

## Deploy the application[¶](#deploy-the-application)

As all the files in the app package are ready, we can start a Vespa instance - here using Docker. Deploy the app package:

In \[8\]:

Copied!

```
from vespa.deployment import VespaDocker

vespa_docker = VespaDocker()
app = vespa_docker.deploy_from_disk(application_name="wiki", application_root="pkg")
```

from vespa.deployment import VespaDocker vespa_docker = VespaDocker() app = vespa_docker.deploy_from_disk(application_name="wiki", application_root="pkg")

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

## Feed documents[¶](#feed-documents)

Download the Wikipedia articles:

In \[9\]:

Copied!

```
! curl -s -H "Accept:application/vnd.github.v3.raw" \
  https://api.github.com/repos/vespa-engine/sample-apps/contents/multi-vector-indexing/ext/articles.jsonl.zst | \
  zstdcat - > articles.jsonl
```

! curl -s -H "Accept:application/vnd.github.v3.raw" \
https://api.github.com/repos/vespa-engine/sample-apps/contents/multi-vector-indexing/ext/articles.jsonl.zst | \
zstdcat - > articles.jsonl

I you do not have ZSTD install, get `articles.jsonl.zip` and unzip it instead.

Feed and index the Wikipedia articles using the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html). As part of feeding, `embed` is called on each article, and the output of this is stored in the `paragraph_embeddings` field:

In \[10\]:

Copied!

```
! vespa config set target local
! vespa feed articles.jsonl
```

! vespa config set target local ! vespa feed articles.jsonl

```
{
  "feeder.seconds": 1.448,
  "feeder.ok.count": 8,
  "feeder.ok.rate": 5.524,
  "feeder.error.count": 0,
  "feeder.inflight.count": 0,
  "http.request.count": 8,
  "http.request.bytes": 12958,
  "http.request.MBps": 0.009,
  "http.exception.count": 0,
  "http.response.count": 8,
  "http.response.bytes": 674,
  "http.response.MBps": 0.000,
  "http.response.error.count": 0,
  "http.response.latency.millis.min": 728,
  "http.response.latency.millis.avg": 834,
  "http.response.latency.millis.max": 1446,
  "http.response.code.counts": {
    "200": 8
  }
}
```

Note that creating embeddings is computationally expensive, but this is a small dataset with only 8 articles, so will be done in a few seconds.

The Vespa instance is now populated with the Wikipedia articles, with generated embeddings, and ready for queries. The next sections have examples of various kinds of queries to run on the dataset.

## Simple retrieve all articles with undefined ranking[¶](#simple-retrieve-all-articles-with-undefined-ranking)

Run a query selecting *all* documents, returning two of them. The rank profile is the built-in `unranked` which means no ranking calculations are done, the results are returned in random order:

In \[ \]:

Copied!

```
from vespa.io import VespaQueryResponse

result: VespaQueryResponse = app.query(
    body={
        "yql": "select * from wiki where true",
        "ranking.profile": "unranked",
        "hits": 2,
    }
)
if not result.is_successful():
    raise ValueError(result.get_json())
if len(result.hits) != 2:
    raise ValueError("Expected 2 hits, got {}".format(len(result.hits)))
print(json.dumps(result.hits, indent=4))
```

from vespa.io import VespaQueryResponse result: VespaQueryResponse = app.query( body={ "yql": "select * from wiki where true", "ranking.profile": "unranked", "hits": 2, } ) if not result.is_successful(): raise ValueError(result.get_json()) if len(result.hits) != 2: raise ValueError("Expected 2 hits, got {}".format(len(result.hits))) print(json.dumps(result.hits, indent=4))

## Traditional keyword search with BM25 ranking on the article level[¶](#traditional-keyword-search-with-bm25-ranking-on-the-article-level)

Run a text-search query and use the [bm25](https://docs.vespa.ai/en/reference/bm25.html) ranking profile configured at the start of this guide: `2*bm25(title) + bm25(paragraphs)`. Here, we use BM25 on the `title` and `paragraph` text fields, giving more weight to matches in title:

In \[ \]:

Copied!

```
result = app.query(
    body={
        "yql": "select * from wiki where userQuery()",
        "query": 24,
        "ranking.profile": "bm25",
        "hits": 2,
    }
)
if len(result.hits) != 2:
    raise ValueError("Expected 2 hits, got {}".format(len(result.hits)))
print(json.dumps(result.hits, indent=4))
```

result = app.query( body={ "yql": "select * from wiki where userQuery()", "query": 24, "ranking.profile": "bm25", "hits": 2, } ) if len(result.hits) != 2: raise ValueError("Expected 2 hits, got {}".format(len(result.hits))) print(json.dumps(result.hits, indent=4))

## Semantic vector search on the paragraph level[¶](#semantic-vector-search-on-the-paragraph-level)

This query creates an embedding of the query "what does 24 mean in the context of railways" and specifies the `semantic` ranking profile: `cos(distance(field,paragraph_embeddings))`. This will hence compute the distance between the vector in the query and the vectors computed when indexing: `"input paragraphs", "embed", "index", "attribute"`:

In \[14\]:

Copied!

```
result = app.query(
    body={
        "yql": "select * from wiki where {targetHits:2}nearestNeighbor(paragraph_embeddings,q)",
        "input.query(q)": "embed(what does 24 mean in the context of railways)",
        "ranking.profile": "semantic",
        "presentation.format.tensors": "short-value",
        "hits": 2,
    }
)
result.hits
if len(result.hits) != 2:
    raise ValueError("Expected 2 hits, got {}".format(len(result.hits)))
print(json.dumps(result.hits, indent=4))
```

result = app.query( body={ "yql": "select * from wiki where {targetHits:2}nearestNeighbor(paragraph_embeddings,q)", "input.query(q)": "embed(what does 24 mean in the context of railways)", "ranking.profile": "semantic", "presentation.format.tensors": "short-value", "hits": 2, } ) result.hits if len(result.hits) != 2: raise ValueError("Expected 2 hits, got {}".format(len(result.hits))) print(json.dumps(result.hits, indent=4))

```
[
    {
        "id": "id:wikipedia:wiki::9985",
        "relevance": 0.8807156260391702,
        "source": "wiki_content",
        "fields": {
            "matchfeatures": {
                "closest(paragraph_embeddings)": {
                    "4": 1.0
                }
            },
            "sddocname": "wiki",
            "paragraphs": [
                "The 24-hour clock is a way of telling the time in which the day runs from midnight to midnight and is divided into 24 hours, numbered from 0 to 23. It does not use a.m. or p.m. This system is also referred to (only in the US and the English speaking parts of Canada) as military time or (only in the United Kingdom and now very rarely) as continental time. In some parts of the world, it is called railway time. Also, the international standard notation of time (ISO 8601) is based on this format.",
                "A time in the 24-hour clock is written in the form hours:minutes (for example, 01:23), or hours:minutes:seconds (01:23:45). Numbers under 10 have a zero in front (called a leading zero); e.g. 09:07. Under the 24-hour clock system, the day begins at midnight, 00:00, and the last minute of the day begins at 23:59 and ends at 24:00, which is identical to 00:00 of the following day. 12:00 can only be mid-day. Midnight is called 24:00 and is used to mean the end of the day and 00:00 is used to mean the beginning of the day. For example, you would say \"Tuesday at 24:00\" and \"Wednesday at 00:00\" to mean exactly the same time.",
                "However, the US military prefers not to say 24:00 - they do not like to have two names for the same thing, so they always say \"23:59\", which is one minute before midnight.",
                "24-hour clock time is used in computers, military, public safety, and transport. In many Asian, European and Latin American countries people use it to write the time. Many European people use it in speaking.",
                "In railway timetables 24:00 means the \"end\" of the day. For example, a train due to arrive at a station during the last minute of a day arrives at 24:00; but trains which depart during the first minute of the day go at 00:00."
            ],
            "documentid": "id:wikipedia:wiki::9985",
            "title": "24-hour clock",
            "url": "https://simple.wikipedia.org/wiki?curid=9985"
        }
    },
    {
        "id": "id:wikipedia:wiki::59079",
        "relevance": 0.7972394509946005,
        "source": "wiki_content",
        "fields": {
            "matchfeatures": {
                "closest(paragraph_embeddings)": {
                    "4": 1.0
                }
            },
            "sddocname": "wiki",
            "paragraphs": [
                "Logic gates are digital components. They normally work at only two levels of voltage, a positive level and zero level. Commonly they work based on two states: \"On\" and \"Off\". In the On state, voltage is positive. In the Off state, the voltage is at zero. The On state usually uses a voltage in the range of 3.5 to 5 volts. This range can be lower for some uses.",
                "Logic gates compare the state at their inputs to decide what the state at their output should be. A logic gate is \"on\" or active when its rules are correctly met. At this time, electricity is flowing through the gate and the voltage at its output is at the level of its On state.",
                "Logic gates are electronic versions of Boolean logic. Truth tables will tell you what the output will be, depending on the inputs.",
                "AND gates have two inputs. The output of an AND gate is on only if both inputs are on. If at least one of the inputs is off, the output will be off.",
                "Using the image at the right, if \"A\" and \"B\" are both in an On state, the output (out) will be an On state. If either \"A\" or \"B\" is in an Off state, the output will also be in an Off state. \"A\" and \"B\" must be On for the output to be On.",
                "OR gates have two inputs. The output of an OR gate will be on if at least one of the inputs are on. If both inputs are off, the output will be off.",
                "Using the image at the right, if either \"A\" or \"B\" is On, the output (\"out\") will also be On. If both \"A\" and \"B\" are Off, the output will be Off.",
                "The NOT logic gate has only one input. If the input is On then the output will be Off. In other words, the NOT logic gate changes the signal from On to Off or from Off to On. It is sometimes called an inverter.",
                "XOR (\"exclusive or\") gates have two inputs. The output of a XOR gate will be true only if the two inputs are different from each other. If both inputs are the same, the output will be off.",
                "NAND means not both. It is called NAND because it means \"not and.\" This means that it will always output true unless both inputs are on.",
                "XNOR means \"not exclusive or.\" This means that it will only output true if both inputs are the same. It is the opposite of a XOR logic gate."
            ],
            "documentid": "id:wikipedia:wiki::59079",
            "title": "Logic gate",
            "url": "https://simple.wikipedia.org/wiki?curid=59079"
        }
    }
]
```

An interesting question then is, of the paragraphs in the document, which one was the closest? When analysing ranking, using [match-features](https://docs.vespa.ai/en/reference/schema-reference.html#match-features) lets you export the scores used in the ranking calculations, see [closest](<https://docs.vespa.ai/en/reference/rank-features.html#closest(name)>) - from the result above:

```
 "matchfeatures": {
                "closest(paragraph_embeddings)": {
                    "4": 1.0
                }
}
```

This means, the tensor of index 4 has the closest match. With this, it is straight forward to feed articles with an array of paragraphs and highlight the best matching paragraph in the document!

In \[17\]:

Copied!

```
def find_best_paragraph(hit: dict) -> str:
    paragraphs = hit["fields"]["paragraphs"]
    match_features = hit["fields"]["matchfeatures"]
    index = int(list(match_features["closest(paragraph_embeddings)"].keys())[0])
    return paragraphs[index]
```

def find_best_paragraph(hit: dict) -> str: paragraphs = hit["fields"]["paragraphs"] match_features = hit["fields"]["matchfeatures"] index = int(list(match_features["closest(paragraph_embeddings)"].keys())[0]) return paragraphs[index]

In \[18\]:

Copied!

```
find_best_paragraph(result.hits[0])
```

find_best_paragraph(result.hits[0])

Out\[18\]:

```
'In railway timetables 24:00 means the "end" of the day. For example, a train due to arrive at a station during the last minute of a day arrives at 24:00; but trains which depart during the first minute of the day go at 00:00.'
```

## Hybrid search and ranking[¶](#hybrid-search-and-ranking)

Hybrid combining keyword search on the article level with vector search in the paragraph index:

In \[20\]:

Copied!

```
result = app.query(
    body={
        "yql": "select * from wiki where userQuery() or ({targetHits:1}nearestNeighbor(paragraph_embeddings,q))",
        "input.query(q)": "embed(what does 24 mean in the context of railways)",
        "query": "what does 24 mean in the context of railways",
        "ranking.profile": "hybrid",
        "presentation.format.tensors": "short-value",
        "hits": 1,
    }
)
if len(result.hits) != 1:
    raise ValueError("Expected 1 hits, got {}".format(len(result.hits)))
print(json.dumps(result.hits, indent=4))
```

result = app.query( body={ "yql": "select * from wiki where userQuery() or ({targetHits:1}nearestNeighbor(paragraph_embeddings,q))", "input.query(q)": "embed(what does 24 mean in the context of railways)", "query": "what does 24 mean in the context of railways", "ranking.profile": "hybrid", "presentation.format.tensors": "short-value", "hits": 1, } ) if len(result.hits) != 1: raise ValueError("Expected 1 hits, got {}".format(len(result.hits))) print(json.dumps(result.hits, indent=4))

```
[
    {
        "id": "id:wikipedia:wiki::9985",
        "relevance": 4.163399168193791,
        "source": "wiki_content",
        "fields": {
            "matchfeatures": {
                "bm25(paragraphs)": 10.468827250036052,
                "bm25(title)": 1.1272217840066168,
                "closest(paragraph_embeddings)": {
                    "4": 1.0
                },
                "firstPhase": 0.8807156260391702,
                "all_paragraph_similarities": {
                    "1": 0.8030083179473877,
                    "2": 0.7992785573005676,
                    "3": 0.8273358345031738,
                    "4": 0.8807156085968018,
                    "0": 0.849757194519043
                },
                "avg_paragraph_similarity": 0.8320191025733947,
                "max_paragraph_similarity": 0.8807156085968018
            },
            "sddocname": "wiki",
            "paragraphs": [
                "<hi>The</hi> <hi>24</hi>-hour clock is a way <hi>of</hi> telling <hi>the</hi> time <hi>in</hi> which <hi>the</hi> day runs from midnight to midnight and is divided into <hi>24</hi> hours, numbered from 0 to 23. It <hi>does</hi> not use a.m. or p.m. This system is also referred to (only <hi>in</hi> <hi>the</hi> US and <hi>the</hi> English speaking parts <hi>of</hi> Canada) as military time or (only <hi>in</hi> <hi>the</hi> United Kingdom and now very rarely) as continental time. <hi>In</hi> some parts <hi>of</hi> <hi>the</hi> world, it is called <hi>railway</hi> time. Also, <hi>the</hi> international standard notation <hi>of</hi> time (ISO 8601) is based on this format.",
                "A time <hi>in</hi> <hi>the</hi> <hi>24</hi>-hour clock is written <hi>in</hi> <hi>the</hi> form hours:minutes (for example, 01:23), or hours:minutes:seconds (01:23:45). Numbers under 10 have a zero <hi>in</hi> front (called a leading zero); e.g. 09:07. Under <hi>the</hi> <hi>24</hi>-hour clock system, <hi>the</hi> day begins at midnight, 00:00, and <hi>the</hi> last minute <hi>of</hi> <hi>the</hi> day begins at 23:59 and ends at <hi>24</hi>:00, which is identical to 00:00 <hi>of</hi> <hi>the</hi> following day. 12:00 can only be mid-day. Midnight is called <hi>24</hi>:00 and is used to <hi>mean</hi> <hi>the</hi> end <hi>of</hi> <hi>the</hi> day and 00:00 is used to <hi>mean</hi> <hi>the</hi> beginning <hi>of</hi> <hi>the</hi> day. For example, you would say \"Tuesday at <hi>24</hi>:00\" and \"Wednesday at 00:00\" to <hi>mean</hi> exactly <hi>the</hi> same time.",
                "However, <hi>the</hi> US military prefers not to say <hi>24</hi>:00 - they <hi>do</hi> not like to have two names for <hi>the</hi> same thing, so they always say \"23:59\", which is one minute before midnight.",
                "<hi>24</hi>-hour clock time is used <hi>in</hi> computers, military, public safety, and transport. <hi>In</hi> many Asian, European and Latin American countries people use it to write <hi>the</hi> time. Many European people use it <hi>in</hi> speaking.",
                "<hi>In</hi> <hi>railway</hi> timetables <hi>24</hi>:00 means <hi>the</hi> \"end\" <hi>of</hi> <hi>the</hi> day. For example, a train due to arrive at a station during <hi>the</hi> last minute <hi>of</hi> a day arrives at <hi>24</hi>:00; but trains which depart during <hi>the</hi> first minute <hi>of</hi> <hi>the</hi> day go at 00:00."
            ],
            "documentid": "id:wikipedia:wiki::9985",
            "title": "24-hour clock",
            "url": "https://simple.wikipedia.org/wiki?curid=9985"
        }
    }
]
```

This case combines exact search with nearestNeighbor search. The `hybrid` rank-profile above also calculates several additional features using [tensor expressions](https://docs.vespa.ai/en/tensor-user-guide.html):

- `firstPhase` is the score of the first ranking phase, configured in the hybrid profile as `cos(distance(field, paragraph_embeddings))`.
- `all_paragraph_similarities` returns all the similarity scores for all paragraphs.
- `avg_paragraph_similarity` is the average similarity score across all the paragraphs.
- `max_paragraph_similarity` is the same as `firstPhase`, but computed using a tensor expression.

These additional features are calculated during [second-phase ranking](https://docs.vespa.ai/en/phased-ranking.html) to limit the number of vector computations.

The [Tensor Playground](https://docs.vespa.ai/playground/) is useful to play with tensor expressions.

The [Hybrid Search](https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa/) blog post series is a good read to learn more about hybrid ranking!

In \[23\]:

Copied!

```
def find_paragraph_scores(hit: dict) -> str:
    paragraphs = hit["fields"]["paragraphs"]
    match_features = hit["fields"]["matchfeatures"]
    indexes = [int(v) for v in match_features["all_paragraph_similarities"]]
    scores = list(match_features["all_paragraph_similarities"].values())
    return list(zip([paragraphs[i] for i in indexes], scores))
```

def find_paragraph_scores(hit: dict) -> str: paragraphs = hit["fields"]["paragraphs"] match_features = hit["fields"]["matchfeatures"] indexes = \[int(v) for v in match_features["all_paragraph_similarities"]\] scores = list(match_features["all_paragraph_similarities"].values()) return list(zip(\[paragraphs[i] for i in indexes\], scores))

In \[24\]:

Copied!

```
find_paragraph_scores(result.hits[0])
```

find_paragraph_scores(result.hits[0])

Out\[24\]:

```
[('A time <hi>in</hi> <hi>the</hi> <hi>24</hi>-hour clock is written <hi>in</hi> <hi>the</hi> form hours:minutes (for example, 01:23), or hours:minutes:seconds (01:23:45). Numbers under 10 have a zero <hi>in</hi> front (called a leading zero); e.g. 09:07. Under <hi>the</hi> <hi>24</hi>-hour clock system, <hi>the</hi> day begins at midnight, 00:00, and <hi>the</hi> last minute <hi>of</hi> <hi>the</hi> day begins at 23:59 and ends at <hi>24</hi>:00, which is identical to 00:00 <hi>of</hi> <hi>the</hi> following day. 12:00 can only be mid-day. Midnight is called <hi>24</hi>:00 and is used to <hi>mean</hi> <hi>the</hi> end <hi>of</hi> <hi>the</hi> day and 00:00 is used to <hi>mean</hi> <hi>the</hi> beginning <hi>of</hi> <hi>the</hi> day. For example, you would say "Tuesday at <hi>24</hi>:00" and "Wednesday at 00:00" to <hi>mean</hi> exactly <hi>the</hi> same time.',
  0.8030083179473877),
 ('However, <hi>the</hi> US military prefers not to say <hi>24</hi>:00 - they <hi>do</hi> not like to have two names for <hi>the</hi> same thing, so they always say "23:59", which is one minute before midnight.',
  0.7992785573005676),
 ('<hi>24</hi>-hour clock time is used <hi>in</hi> computers, military, public safety, and transport. <hi>In</hi> many Asian, European and Latin American countries people use it to write <hi>the</hi> time. Many European people use it <hi>in</hi> speaking.',
  0.8273358345031738),
 ('<hi>In</hi> <hi>railway</hi> timetables <hi>24</hi>:00 means <hi>the</hi> "end" <hi>of</hi> <hi>the</hi> day. For example, a train due to arrive at a station during <hi>the</hi> last minute <hi>of</hi> a day arrives at <hi>24</hi>:00; but trains which depart during <hi>the</hi> first minute <hi>of</hi> <hi>the</hi> day go at 00:00.',
  0.8807156085968018),
 ('<hi>The</hi> <hi>24</hi>-hour clock is a way <hi>of</hi> telling <hi>the</hi> time <hi>in</hi> which <hi>the</hi> day runs from midnight to midnight and is divided into <hi>24</hi> hours, numbered from 0 to 23. It <hi>does</hi> not use a.m. or p.m. This system is also referred to (only <hi>in</hi> <hi>the</hi> US and <hi>the</hi> English speaking parts <hi>of</hi> Canada) as military time or (only <hi>in</hi> <hi>the</hi> United Kingdom and now very rarely) as continental time. <hi>In</hi> some parts <hi>of</hi> <hi>the</hi> world, it is called <hi>railway</hi> time. Also, <hi>the</hi> international standard notation <hi>of</hi> time (ISO 8601) is based on this format.',
  0.849757194519043)]
```

## Hybrid search and filter[¶](#hybrid-search-and-filter)

YQL is a structured query langauge. In the query examples, the user input is fed as-is using the `userQuery()` operator.

Filters are normally separate from the user input, below is an example of adding a filter `url contains "9985"` to the YQL string.

Finally, the use the [Query API](https://docs.vespa.ai/en/query-api.html) for other options, like highlighting - here disable [bolding](https://docs.vespa.ai/en/reference/schema-reference.html#bolding):

In \[25\]:

Copied!

```
result = app.query(
    body={
        "yql": 'select * from wiki where url contains "9985" and userQuery() or ({targetHits:1}nearestNeighbor(paragraph_embeddings,q))',
        "input.query(q)": "embed(what does 24 mean in the context of railways)",
        "query": "what does 24 mean in the context of railways",
        "ranking.profile": "hybrid",
        "bolding": False,
        "presentation.format.tensors": "short-value",
        "hits": 1,
    }
)
if len(result.hits) != 1:
    raise ValueError("Expected one hit, got {}".format(len(result.hits)))
print(json.dumps(result.hits, indent=4))
```

result = app.query( body={ "yql": 'select * from wiki where url contains "9985" and userQuery() or ({targetHits:1}nearestNeighbor(paragraph_embeddings,q))', "input.query(q)": "embed(what does 24 mean in the context of railways)", "query": "what does 24 mean in the context of railways", "ranking.profile": "hybrid", "bolding": False, "presentation.format.tensors": "short-value", "hits": 1, } ) if len(result.hits) != 1: raise ValueError("Expected one hit, got {}".format(len(result.hits))) print(json.dumps(result.hits, indent=4))

```
[
    {
        "id": "id:wikipedia:wiki::9985",
        "relevance": 4.307079208249452,
        "source": "wiki_content",
        "fields": {
            "matchfeatures": {
                "bm25(paragraphs)": 10.468827250036052,
                "bm25(title)": 1.1272217840066168,
                "closest(paragraph_embeddings)": {
                    "type": "tensor<float>(p{})",
                    "cells": {
                        "4": 1.0
                    }
                },
                "firstPhase": 0.8807156260391702,
                "all_paragraph_similarities": {
                    "type": "tensor<float>(p{})",
                    "cells": {
                        "1": 0.8030083179473877,
                        "2": 0.7992785573005676,
                        "3": 0.8273358345031738,
                        "4": 0.8807156085968018,
                        "0": 0.849757194519043
                    }
                },
                "avg_paragraph_similarity": 0.8320191025733947,
                "max_paragraph_similarity": 0.8807156085968018
            },
            "sddocname": "wiki",
            "paragraphs": [
                "The 24-hour clock is a way of telling the time in which the day runs from midnight to midnight and is divided into 24 hours, numbered from 0 to 23. It does not use a.m. or p.m. This system is also referred to (only in the US and the English speaking parts of Canada) as military time or (only in the United Kingdom and now very rarely) as continental time. In some parts of the world, it is called railway time. Also, the international standard notation of time (ISO 8601) is based on this format.",
                "A time in the 24-hour clock is written in the form hours:minutes (for example, 01:23), or hours:minutes:seconds (01:23:45). Numbers under 10 have a zero in front (called a leading zero); e.g. 09:07. Under the 24-hour clock system, the day begins at midnight, 00:00, and the last minute of the day begins at 23:59 and ends at 24:00, which is identical to 00:00 of the following day. 12:00 can only be mid-day. Midnight is called 24:00 and is used to mean the end of the day and 00:00 is used to mean the beginning of the day. For example, you would say \"Tuesday at 24:00\" and \"Wednesday at 00:00\" to mean exactly the same time.",
                "However, the US military prefers not to say 24:00 - they do not like to have two names for the same thing, so they always say \"23:59\", which is one minute before midnight.",
                "24-hour clock time is used in computers, military, public safety, and transport. In many Asian, European and Latin American countries people use it to write the time. Many European people use it in speaking.",
                "In railway timetables 24:00 means the \"end\" of the day. For example, a train due to arrive at a station during the last minute of a day arrives at 24:00; but trains which depart during the first minute of the day go at 00:00."
            ],
            "documentid": "id:wikipedia:wiki::9985",
            "title": "24-hour clock",
            "url": "https://simple.wikipedia.org/wiki?curid=9985"
        }
    }
]
```

In short, the above query demonstrates how easy it is to combine various ranking strategies, and also combine with filters.

To learn more about pre-filtering vs post-filtering, read [Filtering strategies and serving performance](https://blog.vespa.ai/constrained-approximate-nearest-neighbor-search/). [Semantic search with multi-vector indexing](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/) is a great read overall for this domain.

## Cleanup[¶](#cleanup)

In \[26\]:

Copied!

```
vespa_docker.container.stop()
vespa_docker.container.remove()
```

vespa_docker.container.stop() vespa_docker.container.remove()
