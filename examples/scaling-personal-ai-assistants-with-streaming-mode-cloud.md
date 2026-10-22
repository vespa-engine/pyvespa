# Building cost-efficient retrieval-augmented personal AI assistants[¶](#building-cost-efficient-retrieval-augmented-personal-ai-assistants)

This notebook demonstrates how to use [Vespa streaming mode](https://docs.vespa.ai/en/streaming-search.html) for cost-efficient retrieval for applications that store and retrieve personal data. You can read more about Vespa vector streaming search in these two blog posts:

- [Announcing vector streaming search: AI assistants at scale without breaking the bank](https://blog.vespa.ai/announcing-vector-streaming-search/)
- [Yahoo Mail turns to Vespa to do RAG at scale](https://blog.vespa.ai/yahoo-mail-turns-to-vespa-to-do-rag-at-scale/)

## A summary of Vespa streaming mode[¶](#a-summary-of-vespa-streaming-mode)

Vespa’s streaming search solution lets you make the user id a part of the document ID so that Vespa can use it to co-locate the data of each user on a small set of nodes and the same chunk of disk. This allows you to do searches over a user’s data with low latency without keeping any user’s data in memory or paying the cost of managing indexes.

- There is no accuracy drop for vector search as it uses exact vector search
- Several orders of magnitude higher throughput (No expensive index builds to support approximate search)
- Documents (including vector data) are disk-based.
- Ultra-low memory requirements (fixed per document)

This notebook connects a custom [LlamaIndex](https://docs.llamaindex.ai/) [Retriever](https://docs.llamaindex.ai/) with a [Vespa](https://vespa.ai/) app using streaming mode to retrieve personal data. The focus is on how to use the streaming mode feature.

First, install dependencies:

In \[ \]:

Copied!

```
!pip3 install -U pyvespa llama-index vespacli
```

!pip3 install -U pyvespa llama-index vespacli

## Synthetic Mail & Calendar Data[¶](#synthetic-mail-calendar-data)

There are few public email datasets because people care about their privacy, so this notebook uses synthetic data to examine how to use Vespa streaming mode. We create two generator functions that returns Python `dict`s with synthetic mail and calendar data.

Notice that the dict has three keys:

- `id`
- `groupname`
- `fields`

This is the expected feed format for [PyVespa](https://vespa-engine.github.io/pyvespa/reads-writes.md) feed operations and where PyVespa will use these to build a Vespa [document v1 API](https://docs.vespa.ai/en/document-v1-api-guide.html) request(s). The `groupname` key is only to be used when using streaming mode.

In \[2\]:

Copied!

```
from typing import List


def synthetic_mail_data_generator() -> List[dict]:
    synthetic_mails = [
        {
            "id": 1,
            "groupname": "bergum@vespa.ai",
            "fields": {
                "subject": "LlamaIndex news, 2023-11-14",
                "to": "bergum@vespa.ai",
                "body": """Hello Llama Friends 🦙 LlamaIndex is 1 year old this week! 🎉 To celebrate, we're taking a stroll down memory 
                    lane on our blog with twelve milestones from our first year. Be sure to check it out.""",
                "from": "news@llamaindex.ai",
                "display_date": "2023-11-15T09:00:00Z",
            },
        },
        {
            "id": 2,
            "groupname": "bergum@vespa.ai",
            "fields": {
                "subject": "Dentist Appointment Reminder",
                "to": "bergum@vespa.ai",
                "body": "Dear Jo Kristian ,\nThis is a reminder for your upcoming dentist appointment on 2023-12-04 at 09:30. Please arrive 15 minutes early.\nBest regards,\nDr. Dentist",
                "from": "dentist@dentist.no",
                "display_date": "2023-11-15T15:30:00Z",
            },
        },
        {
            "id": 1,
            "groupname": "giraffe@wildlife.ai",
            "fields": {
                "subject": "Wildlife Update: Giraffe Edition",
                "to": "giraffe@wildlife.ai",
                "body": "Dear Wildlife Enthusiasts 🦒, We're thrilled to share the latest insights into giraffe behavior in the wild. Join us on an adventure as we explore their natural habitat and learn more about these majestic creatures.",
                "from": "updates@wildlife.ai",
                "display_date": "2023-11-12T14:30:00Z",
            },
        },
        {
            "id": 1,
            "groupname": "penguin@antarctica.ai",
            "fields": {
                "subject": "Antarctica Expedition: Penguin Chronicles",
                "to": "penguin@antarctica.ai",
                "body": "Greetings Explorers 🐧, Our team is embarking on an exciting expedition to Antarctica to study penguin colonies. Stay tuned for live updates and behind-the-scenes footage as we dive into the world of these fascinating birds.",
                "from": "expedition@antarctica.ai",
                "display_date": "2023-11-11T11:45:00Z",
            },
        },
        {
            "id": 1,
            "groupname": "space@exploration.ai",
            "fields": {
                "subject": "Space Exploration News: November Edition",
                "to": "space@exploration.ai",
                "body": "Hello Space Enthusiasts 🚀, Join us as we highlight the latest discoveries and breakthroughs in space exploration. From distant galaxies to new technologies, there's a lot to explore!",
                "from": "news@exploration.ai",
                "display_date": "2023-11-01T16:20:00Z",
            },
        },
        {
            "id": 1,
            "groupname": "ocean@discovery.ai",
            "fields": {
                "subject": "Ocean Discovery: Hidden Treasures Unveiled",
                "to": "ocean@discovery.ai",
                "body": "Dear Ocean Explorers 🌊, Dive deep into the secrets of the ocean with our latest discoveries. From undiscovered species to underwater landscapes, our team is uncovering the wonders of the deep blue.",
                "from": "discovery@ocean.ai",
                "display_date": "2023-10-01T10:15:00Z",
            },
        },
    ]
    for mail in synthetic_mails:
        yield mail
```

from typing import List def synthetic_mail_data_generator() -> List\[dict\]: synthetic_mails = [ { "id": 1, "groupname": "bergum@vespa.ai", "fields": { "subject": "LlamaIndex news, 2023-11-14", "to": "bergum@vespa.ai", "body": """Hello Llama Friends 🦙 LlamaIndex is 1 year old this week! 🎉 To celebrate, we're taking a stroll down memory lane on our blog with twelve milestones from our first year. Be sure to check it out.""", "from": "news@llamaindex.ai", "display_date": "2023-11-15T09:00:00Z", }, }, { "id": 2, "groupname": "bergum@vespa.ai", "fields": { "subject": "Dentist Appointment Reminder", "to": "bergum@vespa.ai", "body": "Dear Jo Kristian ,\\nThis is a reminder for your upcoming dentist appointment on 2023-12-04 at 09:30. Please arrive 15 minutes early.\\nBest regards,\\nDr. Dentist", "from": "dentist@dentist.no", "display_date": "2023-11-15T15:30:00Z", }, }, { "id": 1, "groupname": "giraffe@wildlife.ai", "fields": { "subject": "Wildlife Update: Giraffe Edition", "to": "giraffe@wildlife.ai", "body": "Dear Wildlife Enthusiasts 🦒, We're thrilled to share the latest insights into giraffe behavior in the wild. Join us on an adventure as we explore their natural habitat and learn more about these majestic creatures.", "from": "updates@wildlife.ai", "display_date": "2023-11-12T14:30:00Z", }, }, { "id": 1, "groupname": "penguin@antarctica.ai", "fields": { "subject": "Antarctica Expedition: Penguin Chronicles", "to": "penguin@antarctica.ai", "body": "Greetings Explorers 🐧, Our team is embarking on an exciting expedition to Antarctica to study penguin colonies. Stay tuned for live updates and behind-the-scenes footage as we dive into the world of these fascinating birds.", "from": "expedition@antarctica.ai", "display_date": "2023-11-11T11:45:00Z", }, }, { "id": 1, "groupname": "space@exploration.ai", "fields": { "subject": "Space Exploration News: November Edition", "to": "space@exploration.ai", "body": "Hello Space Enthusiasts 🚀, Join us as we highlight the latest discoveries and breakthroughs in space exploration. From distant galaxies to new technologies, there's a lot to explore!", "from": "news@exploration.ai", "display_date": "2023-11-01T16:20:00Z", }, }, { "id": 1, "groupname": "ocean@discovery.ai", "fields": { "subject": "Ocean Discovery: Hidden Treasures Unveiled", "to": "ocean@discovery.ai", "body": "Dear Ocean Explorers 🌊, Dive deep into the secrets of the ocean with our latest discoveries. From undiscovered species to underwater landscapes, our team is uncovering the wonders of the deep blue.", "from": "discovery@ocean.ai", "display_date": "2023-10-01T10:15:00Z", }, }, ] for mail in synthetic_mails: yield mail

In \[3\]:

Copied!

```
from typing import List


def synthetic_calendar_data_generator() -> List[dict]:
    calendar_data = [
        {
            "id": 1,
            "groupname": "bergum@vespa.ai",
            "fields": {
                "subject": "Dentist Appointment",
                "to": "bergum@vespa.ai",
                "body": "Dentist appointment at 2023-12-04 at 09:30 - 1 hour duration",
                "from": "dentist@dentist.no",
                "display_date": "2023-11-15T15:30:00Z",
                "duration": 60,
            },
        },
        {
            "id": 2,
            "groupname": "bergum@vespa.ai",
            "fields": {
                "subject": "Public Cloud Platform Events",
                "to": "bergum@vespa.ai",
                "body": "The cloud team continues to push new features and improvements to the platform. Join us for a live demo of the latest updates",
                "from": "public-cloud-platform-events",
                "display_date": "2023-11-21T09:30:00Z",
                "duration": 60,
            },
        },
    ]
    for event in calendar_data:
        yield event
```

from typing import List def synthetic_calendar_data_generator() -> List\[dict\]: calendar_data = [ { "id": 1, "groupname": "bergum@vespa.ai", "fields": { "subject": "Dentist Appointment", "to": "bergum@vespa.ai", "body": "Dentist appointment at 2023-12-04 at 09:30 - 1 hour duration", "from": "dentist@dentist.no", "display_date": "2023-11-15T15:30:00Z", "duration": 60, }, }, { "id": 2, "groupname": "bergum@vespa.ai", "fields": { "subject": "Public Cloud Platform Events", "to": "bergum@vespa.ai", "body": "The cloud team continues to push new features and improvements to the platform. Join us for a live demo of the latest updates", "from": "public-cloud-platform-events", "display_date": "2023-11-21T09:30:00Z", "duration": 60, }, }, ] for event in calendar_data: yield event

## Defining a Vespa application[¶](#defining-a-vespa-application)

[PyVespa](https://vespa-engine.github.io/pyvespa/) help us build the [Vespa application package](https://docs.vespa.ai/en/application-packages.html). A Vespa application package consists of configuration files.

First, we define a [Vespa schema](https://docs.vespa.ai/en/schemas.html). [PyVespa](https://vespa-engine.github.io/pyvespa/) offers a programmatic API for creating the schema. In the end, it is serialized to a file (`<schema>.sd`) before it can be deployed to Vespa.

Vespa is statically typed, so we need to define the fields and their type in the schema before we can start feeding documents.\
Note that we set `mode` to `streaming` which enables [Vespa streaming mode for this schema](https://docs.vespa.ai/en/streaming-search.html). Other valid modes are `indexed` and `store-only`.

In \[4\]:

Copied!

```
from vespa.package import Schema, Document, Field, FieldSet, HNSW

mail_schema = Schema(
    name="mail",
    mode="streaming",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary", "index"]),
            Field(name="subject", type="string", indexing=["summary", "index"]),
            Field(name="to", type="string", indexing=["summary", "index"]),
            Field(name="from", type="string", indexing=["summary", "index"]),
            Field(name="body", type="string", indexing=["summary", "index"]),
            Field(name="display_date", type="string", indexing=["summary"]),
            Field(
                name="timestamp",
                type="long",
                indexing=[
                    "input display_date",
                    "to_epoch_second",
                    "summary",
                    "attribute",
                ],
                is_document_field=False,
            ),
            Field(
                name="embedding",
                type="tensor<bfloat16>(x[384])",
                indexing=[
                    'input subject ." ". input body',
                    "embed e5",
                    "attribute",
                    "index",
                ],
                ann=HNSW(distance_metric="angular"),
                is_document_field=False,
            ),
        ],
    ),
    fieldsets=[FieldSet(name="default", fields=["subject", "body", "to", "from"])],
)
```

from vespa.package import Schema, Document, Field, FieldSet, HNSW mail_schema = Schema( name="mail", mode="streaming", document=Document( fields=\[ Field(name="id", type="string", indexing=["summary", "index"]), Field(name="subject", type="string", indexing=["summary", "index"]), Field(name="to", type="string", indexing=["summary", "index"]), Field(name="from", type="string", indexing=["summary", "index"]), Field(name="body", type="string", indexing=["summary", "index"]), Field(name="display_date", type="string", indexing=["summary"]), Field( name="timestamp", type="long", indexing=[ "input display_date", "to_epoch_second", "summary", "attribute", ], is_document_field=False, ), Field( name="embedding", type="tensor<bfloat16>(x[384])", indexing=[ 'input subject ." ". input body', "embed e5", "attribute", "index", ], ann=HNSW(distance_metric="angular"), is_document_field=False, ), \], ), fieldsets=\[FieldSet(name="default", fields=["subject", "body", "to", "from"])\], )

In the `mail` schema, we have six document fields; these are provided by us when we feed documents of type `mail` to this app. The [fieldset](https://docs.vespa.ai/en/schemas.html#fieldset) defines which fields are matched against when we do not mention explicit field names when querying. We can add as many fieldsets as we like without duplicating content.

In addition to the fields within the `document`, there are two synthetic fields in the schema, `timestamp` and `embedding`, using Vespa [indexing expressions](https://docs.vespa.ai/en/reference/indexing-language-reference.html) taking inputs from the document and performing conversions.

- the `timestamp` field takes the input `display_date` and uses the [to_epoch_second converter](https://docs.vespa.ai/en/reference/indexing-language-reference.html#converters) to convert the display date into an epoch timestamp. This is useful because we can calculate the document's age and use the `freshness(timestamp)` rank feature during ranking phases.
- the `embedding` tensor field takes the subject and body as input and feeds that into an [embed](https://docs.vespa.ai/en/embedding.html#embedding-a-document-field) function that uses an embedding model to map the string input into an embedding vector representation using 384 dimensions with `bfloat16` precision. Vectors in Vespa are represented as [Tensors](https://docs.vespa.ai/en/tensor-user-guide.html).

In \[5\]:

Copied!

```
from vespa.package import Schema, Document, Field, FieldSet, HNSW

calendar_schema = Schema(
    name="calendar",
    inherits="mail",
    mode="streaming",
    document=Document(
        inherits="mail",
        fields=[
            Field(name="duration", type="int", indexing=["summary", "index"]),
            Field(name="guests", type="array<string>", indexing=["summary", "index"]),
            Field(name="location", type="string", indexing=["summary", "index"]),
            Field(name="url", type="string", indexing=["summary", "index"]),
            Field(name="address", type="string", indexing=["summary", "index"]),
        ],
    ),
)
```

from vespa.package import Schema, Document, Field, FieldSet, HNSW calendar_schema = Schema( name="calendar", inherits="mail", mode="streaming", document=Document( inherits="mail", fields=\[ Field(name="duration", type="int", indexing=["summary", "index"]), Field(name="guests", type="array<string>", indexing=["summary", "index"]), Field(name="location", type="string", indexing=["summary", "index"]), Field(name="url", type="string", indexing=["summary", "index"]), Field(name="address", type="string", indexing=["summary", "index"]), \], ), )

The observant reader might have noticed the `e5` argument to the `embed` expression in the above `embedding` field. The `e5` argument references a component of the type [hugging-face-embedder](https://docs.vespa.ai/en/embedding.html#huggingface-embedder). We configure the application package and its name with the `mail` schema and the `e5` embedder component.

In \[6\]:

Copied!

```
from vespa.package import ApplicationPackage, Component, Parameter

vespa_app_name = "assistant"
vespa_application_package = ApplicationPackage(
    name=vespa_app_name,
    schema=[mail_schema, calendar_schema],
    components=[
        Component(
            id="e5",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    name="transformer-model",
                    args={
                        "url": "https://github.com/vespa-engine/sample-apps/raw/master/examples/model-exporting/model/e5-small-v2-int8.onnx"
                    },
                ),
                Parameter(
                    name="tokenizer-model",
                    args={
                        "url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/examples/model-exporting/model/tokenizer.json"
                    },
                ),
                Parameter(
                    name="prepend",
                    args={},
                    children=[
                        Parameter(name="query", args={}, children="query: "),
                        Parameter(name="document", args={}, children="passage: "),
                    ],
                ),
            ],
        )
    ],
)
```

from vespa.package import ApplicationPackage, Component, Parameter vespa_app_name = "assistant" vespa_application_package = ApplicationPackage( name=vespa_app_name, schema=[mail_schema, calendar_schema], components=\[ Component( id="e5", type="hugging-face-embedder", parameters=\[ Parameter( name="transformer-model", args={ "url": "https://github.com/vespa-engine/sample-apps/raw/master/examples/model-exporting/model/e5-small-v2-int8.onnx" }, ), Parameter( name="tokenizer-model", args={ "url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/examples/model-exporting/model/tokenizer.json" }, ), Parameter( name="prepend", args={}, children=[ Parameter(name="query", args={}, children="query: "), Parameter(name="document", args={}, children="passage: "), ], ), \], ) \], )

In the last step, we configure [ranking](https://docs.vespa.ai/en/ranking.html) by adding `rank-profile`'s to the mail schema.

Vespa supports [phased ranking](https://docs.vespa.ai/en/phased-ranking.html) and has a rich set of built-in [rank-features](https://docs.vespa.ai/en/reference/rank-features.html).

Users can also define custom functions with [ranking expressions](https://docs.vespa.ai/en/reference/ranking-expressions.html).

In \[7\]:

Copied!

```
from vespa.package import RankProfile, Function, GlobalPhaseRanking, FirstPhaseRanking

keywords_and_freshness = RankProfile(
    name="default",
    functions=[
        Function(
            name="my_function",
            expression="nativeRank(subject) + nativeRank(body) + freshness(timestamp)",
        )
    ],
    first_phase=FirstPhaseRanking(expression="my_function", rank_score_drop_limit=0.02),
    match_features=[
        "nativeRank(subject)",
        "nativeRank(body)",
        "my_function",
        "freshness(timestamp)",
    ],
)

semantic = RankProfile(
    name="semantic",
    functions=[
        Function(name="cosine", expression="max(0,cos(distance(field, embedding)))")
    ],
    inputs=[("query(q)", "tensor<float>(x[384])"), ("query(threshold)", "", "0.75")],
    first_phase=FirstPhaseRanking(
        expression="if(cosine > query(threshold), cosine, -1)",
        rank_score_drop_limit=0.1,
    ),
    match_features=[
        "cosine",
        "freshness(timestamp)",
        "distance(field, embedding)",
        "query(threshold)",
    ],
)

fusion = RankProfile(
    name="fusion",
    inherits="semantic",
    functions=[
        Function(
            name="keywords_and_freshness",
            expression=" nativeRank(subject) + nativeRank(body) + freshness(timestamp)",
        ),
        Function(name="semantic", expression="cos(distance(field,embedding))"),
    ],
    inputs=[("query(q)", "tensor<float>(x[384])"), ("query(threshold)", "", "0.75")],
    first_phase=FirstPhaseRanking(
        expression="if(cosine > query(threshold), cosine, -1)",
        rank_score_drop_limit=0.1,
    ),
    match_features=[
        "nativeRank(subject)",
        "keywords_and_freshness",
        "freshness(timestamp)",
        "cosine",
        "query(threshold)",
    ],
    global_phase=GlobalPhaseRanking(
        rerank_count=1000,
        expression="reciprocal_rank_fusion(semantic, keywords_and_freshness)",
    ),
)
```

from vespa.package import RankProfile, Function, GlobalPhaseRanking, FirstPhaseRanking keywords_and_freshness = RankProfile( name="default", functions=[ Function( name="my_function", expression="nativeRank(subject) + nativeRank(body) + freshness(timestamp)", ) ], first_phase=FirstPhaseRanking(expression="my_function", rank_score_drop_limit=0.02), match_features=[ "nativeRank(subject)", "nativeRank(body)", "my_function", "freshness(timestamp)", ], ) semantic = RankProfile( name="semantic", functions=[ Function(name="cosine", expression="max(0,cos(distance(field, embedding)))") ], inputs=\[("query(q)", "tensor<float>(x[384])"), ("query(threshold)", "", "0.75")\], first_phase=FirstPhaseRanking( expression="if(cosine > query(threshold), cosine, -1)", rank_score_drop_limit=0.1, ), match_features=[ "cosine", "freshness(timestamp)", "distance(field, embedding)", "query(threshold)", ], ) fusion = RankProfile( name="fusion", inherits="semantic", functions=[ Function( name="keywords_and_freshness", expression=" nativeRank(subject) + nativeRank(body) + freshness(timestamp)", ), Function(name="semantic", expression="cos(distance(field,embedding))"), ], inputs=\[("query(q)", "tensor<float>(x[384])"), ("query(threshold)", "", "0.75")\], first_phase=FirstPhaseRanking( expression="if(cosine > query(threshold), cosine, -1)", rank_score_drop_limit=0.1, ), match_features=[ "nativeRank(subject)", "keywords_and_freshness", "freshness(timestamp)", "cosine", "query(threshold)", ], global_phase=GlobalPhaseRanking( rerank_count=1000, expression="reciprocal_rank_fusion(semantic, keywords_and_freshness)", ), )

The `default` rank profile defines a custom function `my_function` that computes a linear combination of three different features:

- `nativeRank(subject)` Is a text matching feature , scoped to the `subject` field.
- `nativeRank(body)` Same, but scoped to the `body` field.
- `freshness(timestamp)` This is a built-in [rank-feature](https://docs.vespa.ai/en/reference/rank-features.html#freshness) that returns a number that is close to 1 if the timestamp is recent compared to the current query time.

In \[8\]:

Copied!

```
mail_schema.add_rank_profile(keywords_and_freshness)
mail_schema.add_rank_profile(semantic)
mail_schema.add_rank_profile(fusion)
calendar_schema.add_rank_profile(keywords_and_freshness)
calendar_schema.add_rank_profile(semantic)
calendar_schema.add_rank_profile(fusion)
```

mail_schema.add_rank_profile(keywords_and_freshness) mail_schema.add_rank_profile(semantic) mail_schema.add_rank_profile(fusion) calendar_schema.add_rank_profile(keywords_and_freshness) calendar_schema.add_rank_profile(semantic) calendar_schema.add_rank_profile(fusion)

Finally, we have our basic Vespa schema and application package.

We can serialize the representation to application package files. This is handy when we want to start working with production deployments and when we want to manage the application with version control.

In \[9\]:

Copied!

```
import os

application_directory = "my-assistant-vespa-app"
vespa_application_package.to_files(application_directory)


def print_files_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))


print_files_in_directory(application_directory)
```

import os application_directory = "my-assistant-vespa-app" vespa_application_package.to_files(application_directory) def print_files_in_directory(directory): for root, \_, files in os.walk(directory): for file in files: print(os.path.join(root, file)) print_files_in_directory(application_directory)

```
my-assistant-vespa-app/services.xml
my-assistant-vespa-app/schemas/mail.sd
my-assistant-vespa-app/schemas/calendar.sd
my-assistant-vespa-app/search/query-profiles/default.xml
my-assistant-vespa-app/search/query-profiles/types/root.xml
```

## Deploy the application to Vespa Cloud[¶](#deploy-the-application-to-vespa-cloud)

With the configured application, we can deploy it to [Vespa Cloud](https://cloud.vespa.ai/en/).

To deploy the application to Vespa Cloud we need to create a tenant in the Vespa Cloud:

Create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/) (unless you already have one). This step requires a Google or GitHub account, and will start your [free trial](https://cloud.vespa.ai/en/free-trial).

Make note of the tenant name, it is used in the next steps.

> Note: Deployments to dev and perf expire after 7 days of inactivity, i.e., 7 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 7 more days.

In \[15\]:

Copied!

```
from vespa.deployment import VespaCloud

# Replace with your tenant name from the Vespa Cloud Console
tenant_name = "vespa-team"

# Key is only used for CI/CD. Can be removed if logging in interactively
key = os.getenv("VESPA_TEAM_API_KEY", None)
if key is not None:
    key = key.replace(r"\n", "\n")  # To parse key correctly

vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=vespa_app_name,
    key_content=key,  # Key is only used for CI/CD. Can be removed if logging in interactively
    application_package=vespa_application_package,
)
```

from vespa.deployment import VespaCloud

# Replace with your tenant name from the Vespa Cloud Console

tenant_name = "vespa-team"

# Key is only used for CI/CD. Can be removed if logging in interactively

key = os.getenv("VESPA_TEAM_API_KEY", None) if key is not None: key = key.replace(r"\\n", "\\n") # To parse key correctly vespa_cloud = VespaCloud( tenant=tenant_name, application=vespa_app_name, key_content=key, # Key is only used for CI/CD. Can be removed if logging in interactively application_package=vespa_application_package, )

Now deploy the app to Vespa Cloud dev zone. The first deployment typically takes 2 minutes until the endpoint is up.

In \[ \]:

Copied!

```
from vespa.application import Vespa

app: Vespa = vespa_cloud.deploy(disk_folder=application_directory)
```

from vespa.application import Vespa app: Vespa = vespa_cloud.deploy(disk_folder=application_directory)

## Feeding data to Vespa[¶](#feeding-data-to-vespa)

With the app up and running in Vespa Cloud, we can start feeding and querying our data.

We use the [feed_iterable](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa.feed_iterable) API of pyvespa with a custom `callback` that prints the URL and an error if the operation fails.

We pass the `synthetic_*generator()` and call `feed_iterable` with the specific `schema` and `namespace`.

Read more about [Vespa document IDs](https://docs.vespa.ai/en/documents.html#id-scheme).

In \[ \]:

Copied!

```
from vespa.io import VespaResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Error {response.url} : {response.get_json()}")
    else:
        print(f"Success {response.url}")


app.feed_iterable(
    synthetic_mail_data_generator(),
    schema="mail",
    namespace="assistant",
    callback=callback,
)
app.feed_iterable(
    synthetic_calendar_data_generator(),
    schema="calendar",
    namespace="assistant",
    callback=callback,
)
```

from vespa.io import VespaResponse def callback(response: VespaResponse, id: str): if not response.is_successful(): print(f"Error {response.url} : {response.get_json()}") else: print(f"Success {response.url}") app.feed_iterable( synthetic_mail_data_generator(), schema="mail", namespace="assistant", callback=callback, ) app.feed_iterable( synthetic_calendar_data_generator(), schema="calendar", namespace="assistant", callback=callback, )

### Querying data[¶](#querying-data)

Now, we can also query our data. With [streaming mode](https://docs.vespa.ai/en/reference/query-api-reference.html#streaming), we must pass the `groupname` parameter, or the request will fail with an error.

The query request uses the Vespa Query API and the `Vespa.query()` function supports passing any of the Vespa query API parameters.

Read more about querying Vespa in:

- [Vespa Query API](https://docs.vespa.ai/en/query-api.html)
- [Vespa Query API reference](https://docs.vespa.ai/en/reference/query-api-reference.html)
- [Vespa Query Language API (YQL)](https://docs.vespa.ai/en/query-language.html)

Sample query request for `when is my dentist appointment` for the user `bergum@vespa.ai`:

In \[18\]:

Copied!

```
from vespa.io import VespaQueryResponse
import json

response: VespaQueryResponse = app.query(
    yql="select subject, display_date, to from sources mail where userQuery()",
    query="when is my dentist appointment",
    groupname="bergum@vespa.ai",
    ranking="default",
    timeout="2s",
)
assert response.is_successful()
print(json.dumps(response.hits[0], indent=2))
```

from vespa.io import VespaQueryResponse import json response: VespaQueryResponse = app.query( yql="select subject, display_date, to from sources mail where userQuery()", query="when is my dentist appointment", groupname="bergum@vespa.ai", ranking="default", timeout="2s", ) assert response.is_successful() print(json.dumps(response.hits[0], indent=2))

```
{
  "id": "id:assistant:mail:g=bergum@vespa.ai:2",
  "relevance": 1.134783932836458,
  "source": "assistant_content.mail",
  "fields": {
    "matchfeatures": {
      "freshness(timestamp)": 0.9232458847736625,
      "nativeRank(body)": 0.09246780326887034,
      "nativeRank(subject)": 0.11907024479392506,
      "my_function": 1.134783932836458
    },
    "subject": "Dentist Appointment Reminder",
    "to": "bergum@vespa.ai",
    "display_date": "2023-11-15T15:30:00Z"
  }
}
```

For the above query request, Vespa searched the `default` fieldset which we defined in the schema to match against several fields including the body and the subject. The `default` rank-profile calculated the relevance score as the sum of three rank-features: `nativeRank(body)` + `nativeRank(subject)` + `freshness(`timestamp)`, and the result of this computation is the` relevance`score of the hit. In addition, we also asked for Vespa to return`matchfeatures`that are handy for debugging the final`relevance\` score or for feature logging.

Now, we can try the `semantic` ranking profile, using Vespa's support for nearestNeighbor search. This also exemplifies using the configured `e5` embedder to embed the user query into an embedding representation. See [embedding a query text](https://docs.vespa.ai/en/embedding.html#embedding-a-query-text) for more usage examples of using Vespa embedders.

In \[19\]:

Copied!

```
from vespa.io import VespaQueryResponse
import json

response: VespaQueryResponse = app.query(
    yql="select subject, display_date from mail where {targetHits:10}nearestNeighbor(embedding,q)",
    groupname="bergum@vespa.ai",
    ranking="semantic",
    body={
        "input.query(q)": 'embed(e5, "when is my dentist appointment")',
    },
    timeout="2s",
)
assert response.is_successful()
print(json.dumps(response.hits[0], indent=2))
```

from vespa.io import VespaQueryResponse import json response: VespaQueryResponse = app.query( yql="select subject, display_date from mail where {targetHits:10}nearestNeighbor(embedding,q)", groupname="bergum@vespa.ai", ranking="semantic", body={ "input.query(q)": 'embed(e5, "when is my dentist appointment")', }, timeout="2s", ) assert response.is_successful() print(json.dumps(response.hits[0], indent=2))

```
{
  "id": "id:assistant:mail:g=bergum@vespa.ai:2",
  "relevance": 0.9079386507883569,
  "source": "assistant_content.mail",
  "fields": {
    "matchfeatures": {
      "distance(field,embedding)": 0.4324572498488368,
      "freshness(timestamp)": 0.9232457561728395,
      "query(threshold)": 0.75,
      "cosine": 0.9079386507883569
    },
    "subject": "Dentist Appointment Reminder",
    "display_date": "2023-11-15T15:30:00Z"
  }
}
```

## LlamaIndex Retrievers Introduction[¶](#llamaindex-retrievers-introduction)

Now, we have a basic Vespa app using streaming mode. We likely want to use an LLM framework like [LangChain](https://www.langchain.com/) or [LLamaIndex](https://www.llamaindex.ai/) to build an end-to-end assistant. In this example notebook, we use LLamaIndex retrievers.

LlamaIndex [retriever](https://docs.llamaindex.ai/) abstraction allows developers to add custom retrievers that retrieve information in Retrieval Augmented Generation (RAG) pipelines.

For an excellent introduction to LLamaIndex and its concepts, see [LLamaIndex High-Level Concepts](https://docs.llamaindex.ai/).

To create a custom LlamaIndex retrieve, we implement a class that inherits from `llama_index.retrievers.BaseRetriever.BaseRetriever` and which implements `_retrieve(query)`.

A simple `PersonalAssistantVespaRetriever` could look like the following:

In \[ \]:

Copied!

```
from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.legacy.callbacks.base import CallbackManager

from vespa.application import Vespa
from vespa.io import VespaQueryResponse

from typing import List, Union, Optional


class PersonalAssistantVespaRetriever(BaseRetriever):
    def __init__(
        self,
        app: Vespa,
        user: str,
        hits: int = 5,
        vespa_rank_profile: str = "default",
        vespa_score_cutoff: float = 0.70,
        sources: List[str] = ["mail"],
        fields: List[str] = ["subject", "body"],
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Sample Retriever for a personal assistant application.
        Args:
        param: app: Vespa application object
        param: user: user id to retrieve documents for (used for Vespa streaming groupname)
        param: hits: number of hits to retrieve from Vespa app
        param: vespa_rank_profile: Vespa rank profile to use
        param: vespa_score_cutoff: Vespa score cutoff to use during first-phase ranking
        param: sources: sources to retrieve documents from
        param: fields: fields to retrieve
        """

        self.app = app
        self.hits = hits
        self.user = user
        self.vespa_rank_profile = vespa_rank_profile
        self.vespa_score_cutoff = vespa_score_cutoff
        self.fields = fields
        self.summary_fields = ",".join(fields)
        self.sources = ",".join(sources)
        super().__init__(callback_manager)

    def _retrieve(self, query: Union[str, QueryBundle]) -> List[NodeWithScore]:
        """Retrieve documents from Vespa application."""
        if isinstance(query, QueryBundle):
            query = query.query_str

        if self.vespa_rank_profile == "default":
            yql: str = f"select {self.summary_fields} from mail where userQuery()"
        else:
            yql = f"select {self.summary_fields} from sources {self.sources} where {{targetHits:10}}nearestNeighbor(embedding,q) or userQuery()"
        vespa_body_request = {
            "yql": yql,
            "query": query,
            "hits": self.hits,
            "ranking.profile": self.vespa_rank_profile,
            "timeout": "2s",
            "input.query(threshold)": self.vespa_score_cutoff,
        }
        if self.vespa_rank_profile != "default":
            vespa_body_request["input.query(q)"] = f'embed(e5, "{query}")'

        with self.app.syncio(connections=1) as session:
            response: VespaQueryResponse = session.query(
                body=vespa_body_request, groupname=self.user
            )
            if not response.is_successful():
                raise ValueError(
                    f"Query request failed: {response.status_code}, response payload: {response.get_json()}"
                )

        nodes: List[NodeWithScore] = []
        for hit in response.hits:
            response_fields: dict = hit.get("fields", {})
            text: str = ""
            for field in response_fields.keys():
                if isinstance(response_fields[field], str) and field in self.fields:
                    text += response_fields[field] + " "
            id = hit["id"]
            #
            doc = TextNode(
                id_=id,
                text=text,
                metadata=response_fields,
            )
            nodes.append(NodeWithScore(node=doc, score=hit["relevance"]))
        return nodes
```

from llama_index.legacy.core.base_retriever import BaseRetriever from llama_index.legacy.schema import NodeWithScore, QueryBundle, TextNode from llama_index.legacy.callbacks.base import CallbackManager from vespa.application import Vespa from vespa.io import VespaQueryResponse from typing import List, Union, Optional class PersonalAssistantVespaRetriever(BaseRetriever): def __init__( self, app: Vespa, user: str, hits: int = 5, vespa_rank_profile: str = "default", vespa_score_cutoff: float = 0.70, sources: List[str] = ["mail"], fields: List[str] = ["subject", "body"], callback_manager: Optional[CallbackManager] = None, ) -> None: """Sample Retriever for a personal assistant application. Args: param: app: Vespa application object param: user: user id to retrieve documents for (used for Vespa streaming groupname) param: hits: number of hits to retrieve from Vespa app param: vespa_rank_profile: Vespa rank profile to use param: vespa_score_cutoff: Vespa score cutoff to use during first-phase ranking param: sources: sources to retrieve documents from param: fields: fields to retrieve """ self.app = app self.hits = hits self.user = user self.vespa_rank_profile = vespa_rank_profile self.vespa_score_cutoff = vespa_score_cutoff self.fields = fields self.summary_fields = ",".join(fields) self.sources = ",".join(sources) super().__init__(callback_manager) def \_retrieve(self, query: Union[str, QueryBundle]) -> List\[NodeWithScore\]: """Retrieve documents from Vespa application.""" if isinstance(query, QueryBundle): query = query.query_str if self.vespa_rank_profile == "default": yql: str = f"select {self.summary_fields} from mail where userQuery()" else: yql = f"select {self.summary_fields} from sources {self.sources} where {{targetHits:10}}nearestNeighbor(embedding,q) or userQuery()" vespa_body_request = { "yql": yql, "query": query, "hits": self.hits, "ranking.profile": self.vespa_rank_profile, "timeout": "2s", "input.query(threshold)": self.vespa_score_cutoff, } if self.vespa_rank_profile != "default": vespa_body_request["input.query(q)"] = f'embed(e5, "{query}")' with self.app.syncio(connections=1) as session: response: VespaQueryResponse = session.query( body=vespa_body_request, groupname=self.user ) if not response.is_successful(): raise ValueError( f"Query request failed: {response.status_code}, response payload: {response.get_json()}" ) nodes: List[NodeWithScore] = [] for hit in response.hits: response_fields: dict = hit.get("fields", {}) text: str = "" for field in response_fields.keys(): if isinstance(response_fields[field], str) and field in self.fields: text += response_fields[field] + " " id = hit["id"]

# 

doc = TextNode( id\_=id, text=text, metadata=response_fields, ) nodes.append(NodeWithScore(node=doc, score=hit["relevance"])) return nodes

The above defines a `PersonalAssistantVespaRetriever` which accepts most importantly a [pyvespa](https://vespa-engine.github.io/pyvespa/) `Vespa` application instance.

The YQL specifies a hybrid retrieval query that retrieves both using embedding-based retrieval (vector search) using Vespa's nearest neighbor search operator in combination with traditional keyword matching.

With the above, we can connect to the running Vespa app and initialize the `PersonalAssistantVespaRetriever` for the user `bergum@vespa.ai`. The `user` argument maps to the [streaming search groupname parameter](https://docs.vespa.ai/en/reference/query-api-reference.html#streaming.groupname).

In \[21\]:

Copied!

```
retriever = PersonalAssistantVespaRetriever(
    app=app, user="bergum@vespa.ai", vespa_rank_profile="default"
)
retriever.retrieve("When is my dentist appointment?")
```

retriever = PersonalAssistantVespaRetriever( app=app, user="bergum@vespa.ai", vespa_rank_profile="default" ) retriever.retrieve("When is my dentist appointment?")

Out\[21\]:

```
[NodeWithScore(node=TextNode(id_='id:assistant:mail:g=bergum@vespa.ai:2', embedding=None, metadata={'matchfeatures': {'freshness(timestamp)': 0.9232454989711935, 'nativeRank(body)': 0.09246780326887034, 'nativeRank(subject)': 0.11907024479392506, 'my_function': 1.1347835470339889}, 'subject': 'Dentist Appointment Reminder', 'body': 'Dear Jo Kristian ,\nThis is a reminder for your upcoming dentist appointment on 2023-12-04 at 09:30. Please arrive 15 minutes early.\nBest regards,\nDr. Dentist'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, hash='269fe208f8d43a967dc683e1c9b832b18ddfb0b2efd801ab7e428620c8163021', text='Dentist Appointment Reminder Dear Jo Kristian ,\nThis is a reminder for your upcoming dentist appointment on 2023-12-04 at 09:30. Please arrive 15 minutes early.\nBest regards,\nDr. Dentist ', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=1.1347835470339889),
 NodeWithScore(node=TextNode(id_='id:assistant:mail:g=bergum@vespa.ai:1', embedding=None, metadata={'matchfeatures': {'freshness(timestamp)': 0.9202362397119341, 'nativeRank(body)': 0.02919821398130037, 'nativeRank(subject)': 1.3512214436142505e-38, 'my_function': 0.9494344536932345}, 'subject': 'LlamaIndex news, 2023-11-14', 'body': "Hello Llama Friends 🦙 LlamaIndex is 1 year old this week! 🎉 To celebrate, we're taking a stroll down memory \n                    lane on our blog with twelve milestones from our first year. Be sure to check it out."}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, hash='5e975eaece761d46956c9d301138f29b5c067d3da32fd013bb79c6ee9c033d3d', text="LlamaIndex news, 2023-11-14 Hello Llama Friends 🦙 LlamaIndex is 1 year old this week! 🎉 To celebrate, we're taking a stroll down memory \n                    lane on our blog with twelve milestones from our first year. Be sure to check it out. ", start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.9494344536932345)]
```

These `NodeWithScore` retrieved `default` rank-profile can then be used for the next steps in a generative chain.

We can also try the `semantic` rank-profile, which has rank-score-drop functionality, allowing us to have a per-query time threshold. Altering the threshold will remove context.

In \[22\]:

Copied!

```
retriever = PersonalAssistantVespaRetriever(
    app=app,
    user="bergum@vespa.ai",
    vespa_rank_profile="semantic",
    vespa_score_cutoff=0.6,
    hits=20,
)
retriever.retrieve("When is my dentist appointment?")
```

retriever = PersonalAssistantVespaRetriever( app=app, user="bergum@vespa.ai", vespa_rank_profile="semantic", vespa_score_cutoff=0.6, hits=20, ) retriever.retrieve("When is my dentist appointment?")

Out\[22\]:

```
[NodeWithScore(node=TextNode(id_='id:assistant:mail:g=bergum@vespa.ai:2', embedding=None, metadata={'matchfeatures': {'distance(field,embedding)': 0.43945494361938975, 'freshness(timestamp)': 0.9232453703703704, 'query(threshold)': 0.6, 'cosine': 0.9049836898369259}, 'subject': 'Dentist Appointment Reminder', 'body': 'Dear Jo Kristian ,\nThis is a reminder for your upcoming dentist appointment on 2023-12-04 at 09:30. Please arrive 15 minutes early.\nBest regards,\nDr. Dentist'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, hash='e89f669e6c9cf64ab6a856d9857915481396e2aa84154951327cd889c23f7c4f', text='Dentist Appointment Reminder Dear Jo Kristian ,\nThis is a reminder for your upcoming dentist appointment on 2023-12-04 at 09:30. Please arrive 15 minutes early.\nBest regards,\nDr. Dentist ', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.9049836898369259),
 NodeWithScore(node=TextNode(id_='id:assistant:mail:g=bergum@vespa.ai:1', embedding=None, metadata={'matchfeatures': {'distance(field,embedding)': 0.69930099954744, 'freshness(timestamp)': 0.9202361111111111, 'query(threshold)': 0.6, 'cosine': 0.7652923088511814}, 'subject': 'LlamaIndex news, 2023-11-14', 'body': "Hello Llama Friends 🦙 LlamaIndex is 1 year old this week! 🎉 To celebrate, we're taking a stroll down memory \n                    lane on our blog with twelve milestones from our first year. Be sure to check it out."}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, hash='cb9b588e5b53dbdd0fbe6f7aadfa689d84a5bea23239293bd299347ee9ecd853', text="LlamaIndex news, 2023-11-14 Hello Llama Friends 🦙 LlamaIndex is 1 year old this week! 🎉 To celebrate, we're taking a stroll down memory \n                    lane on our blog with twelve milestones from our first year. Be sure to check it out. ", start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.7652923088511814)]
```

Create a new retriever with sources including both mail and calendar data:

In \[23\]:

Copied!

```
retriever = PersonalAssistantVespaRetriever(
    app=app,
    user="bergum@vespa.ai",
    vespa_rank_profile="fusion",
    sources=["calendar", "mail"],
    vespa_score_cutoff=0.80,
)
retriever.retrieve("When is my dentist appointment?")
```

retriever = PersonalAssistantVespaRetriever( app=app, user="bergum@vespa.ai", vespa_rank_profile="fusion", sources=["calendar", "mail"], vespa_score_cutoff=0.80, ) retriever.retrieve("When is my dentist appointment?")

Out\[23\]:

```
[NodeWithScore(node=TextNode(id_='id:assistant:calendar:g=bergum@vespa.ai:1', embedding=None, metadata={'matchfeatures': {'freshness(timestamp)': 0.9232447273662552, 'nativeRank(subject)': 0.11907024479392506, 'query(threshold)': 0.8, 'cosine': 0.8872983644178517, 'keywords_and_freshness': 1.1606592237923947}, 'subject': 'Dentist Appointment', 'body': 'Dentist appointment at 2023-12-04 at 09:30 - 1 hour duration'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, hash='b30948011cbe9bbf29135384efbc72f85a6eb65113be0eb9762315a022f11ba1', text='Dentist Appointment Dentist appointment at 2023-12-04 at 09:30 - 1 hour duration ', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.03278688524590164),
 NodeWithScore(node=TextNode(id_='id:assistant:mail:g=bergum@vespa.ai:2', embedding=None, metadata={'matchfeatures': {'freshness(timestamp)': 0.9232447273662552, 'nativeRank(subject)': 0.11907024479392506, 'query(threshold)': 0.8, 'cosine': 0.9049836898369259, 'keywords_and_freshness': 1.1347827754290507}, 'subject': 'Dentist Appointment Reminder', 'body': 'Dear Jo Kristian ,\nThis is a reminder for your upcoming dentist appointment on 2023-12-04 at 09:30. Please arrive 15 minutes early.\nBest regards,\nDr. Dentist'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, hash='21c501ccdc6e4b33d388eefa244c5039a0e1ed4b81e4f038916765e22be24705', text='Dentist Appointment Reminder Dear Jo Kristian ,\nThis is a reminder for your upcoming dentist appointment on 2023-12-04 at 09:30. Please arrive 15 minutes early.\nBest regards,\nDr. Dentist ', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.03278688524590164)]
```

In \[24\]:

Copied!

```
app.query(
    yql="select subject, display_date from calendar where duration > 0",
    ranking="default",
    groupname="bergum@vespa.ai",
    timeout="2s",
).json
```

app.query( yql="select subject, display_date from calendar where duration > 0", ranking="default", groupname="bergum@vespa.ai", timeout="2s", ).json

Out\[24\]:

```
{'root': {'id': 'toplevel',
  'relevance': 1.0,
  'fields': {'totalCount': 2},
  'coverage': {'coverage': 100,
   'documents': 2,
   'full': True,
   'nodes': 1,
   'results': 1,
   'resultsFull': 1},
  'children': [{'id': 'id:assistant:calendar:g=bergum@vespa.ai:2',
    'relevance': 0.987133487654321,
    'source': 'assistant_content.calendar',
    'fields': {'matchfeatures': {'freshness(timestamp)': 0.987133487654321,
      'nativeRank(body)': 0.0,
      'nativeRank(subject)': 0.0,
      'my_function': 0.987133487654321},
     'subject': 'Public Cloud Platform Events',
     'display_date': '2023-11-21T09:30:00Z'}},
   {'id': 'id:assistant:calendar:g=bergum@vespa.ai:1',
    'relevance': 0.9232445987654321,
    'source': 'assistant_content.calendar',
    'fields': {'matchfeatures': {'freshness(timestamp)': 0.9232445987654321,
      'nativeRank(body)': 0.0,
      'nativeRank(subject)': 0.0,
      'my_function': 0.9232445987654321},
     'subject': 'Dentist Appointment',
     'display_date': '2023-11-15T15:30:00Z'}}]}}
```

## Conclusion[¶](#conclusion)

In this notebook, we have demonstrated:

- Configuring and using Vespa's streaming mode
- Using multiple document types and schema to organize our data
- Running embedding inference in Vespa
- Hybrid retrieval techniques - combined with score thresholding to filter irrelevant contexts
- Creating a custom LLamaIndex retriever and connecting it with our Vespa app
- Vespa Cloud deployments to sandbox/dev zone

We can now delete the cloud instance:

In \[ \]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()
