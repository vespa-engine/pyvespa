# Multilingual Hybrid Search with Cohere binary embeddings and Vespa[¶](#multilingual-hybrid-search-with-cohere-binary-embeddings-and-vespa)

Cohere just released a new embedding API supporting binary vectors. Read the announcement in the blog post: [Cohere int8 & binary Embeddings - Scale Your Vector Database to Large Datasets](https://cohere.com/blog/int8-binary-embeddings).

> We are excited to announce that Cohere Embed is the first embedding model that natively supports int8 and binary embeddings.

This notebook demonstrates:

- Building a multilingual search application over a sample of the German split of Wikipedia using [binarized cohere embeddings](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary)
- Indexing multiple binary embeddings per document; without having to split the chunks across multiple retrievable units
- Hybrid search, combining the lexical matching capabilities of Vespa with Cohere binary embeddings
- Re-scoring the binarized vectors for improved accuracy

Install the dependencies:

In \[ \]:

Copied!

```
!pip3 install -U pyvespa cohere==4.57 datasets vespacli
```

!pip3 install -U pyvespa cohere==4.57 datasets vespacli

## Dataset exploration[¶](#dataset-exploration)

Cohere has released a large [Wikipedia dataset](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary)

> This dataset contains the wikimedia/wikipedia dataset dump from 2023-11-01 from Wikipedia in all 300+ languages. The embeddings are provided as int8 and ubinary that allow quick search and reduction of your vector index size up to 32.

In \[ \]:

Copied!

```
from datasets import load_dataset

lang = "de"  # Use the first 10K chunks from the German Wikipedia subset
docs = load_dataset(
    "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
    lang,
    split="train",
    streaming=True,
).take(10000)
```

from datasets import load_dataset lang = "de" # Use the first 10K chunks from the German Wikipedia subset docs = load_dataset( "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary", lang, split="train", streaming=True, ).take(10000)

## Aggregate from chunks to pages[¶](#aggregate-from-chunks-to-pages)

We want to aggregate the chunk \<> vector representations into their natural retrievable unit - a Wikipedia page. We can still search the chunks and the chunk vector representation but retrieve pages instead of chunks. This avoids duplicating page-level metadata like url and title, while still being able to have meaningful semantic search representations. For RAG applications, this also means that we have the full page level context available when we retrieve information for the generative phase.

In \[160\]:

Copied!

```
pages = dict()
for d in docs:
    url = d["url"]
    if url not in pages:
        pages[url] = [d]
    else:
        pages[url].append(d)
```

pages = dict() for d in docs: url = d["url"] if url not in pages: pages[url] = [d] else: pages[url].append(d)

In \[173\]:

Copied!

```
print(len(list(pages.keys())))
```

print(len(list(pages.keys())))

```
1866
```

## Defining the Vespa application[¶](#defining-the-vespa-application)

First, we define a [Vespa schema](https://docs.vespa.ai/en/schemas.html) with the fields we want to store and their type.

We use Vespa's multi-vector indexing support - See [Revolutionizing Semantic Search with Multi-Vector HNSW Indexing in Vespa](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/) for details. Highlights

- language for language-specific [linguistic](https://docs.vespa.ai/en/linguistics.html) processing for keyword search
- Two named multi-vector representations with different precision and in-memory versus off-memory
- The named multi-vector representations holds the chunk-level embeddings
- Chunks is an array of string where we enable BM25
- Metadata for the page (url, title)

In \[174\]:

Copied!

```
from vespa.package import Schema, Document, Field, FieldSet

my_schema = Schema(
    name="page",
    mode="index",
    document=Document(
        fields=[
            Field(name="doc_id", type="string", indexing=["summary"]),
            Field(
                name="language",
                type="string",
                indexing=["summary", "index", "set_language"],
                match=["word"],
                rank="filter",
            ),
            Field(
                name="title",
                type="string",
                indexing=["summary", "index"],
                index="enable-bm25",
            ),
            Field(
                name="chunks",
                type="array<string>",
                indexing=["summary", "index"],
                index="enable-bm25",
            ),
            Field(
                name="url",
                type="string",
                indexing=["summary", "index"],
                index="enable-bm25",
            ),
            Field(
                name="binary_vectors",
                type="tensor<int8>(chunk{}, x[128])",
                indexing=["attribute", "index"],
                attribute=["distance-metric: hamming"],
            ),
            Field(
                name="int8_vectors",
                type="tensor<int8>(chunk{}, x[1024])",
                indexing=["attribute"],
                attribute=["paged"],
            ),
        ]
    ),
    fieldsets=[FieldSet(name="default", fields=["chunks", "title"])],
)
```

from vespa.package import Schema, Document, Field, FieldSet my_schema = Schema( name="page", mode="index", document=Document( fields=\[ Field(name="doc_id", type="string", indexing=["summary"]), Field( name="language", type="string", indexing=["summary", "index", "set_language"], match=["word"], rank="filter", ), Field( name="title", type="string", indexing=["summary", "index"], index="enable-bm25", ), Field( name="chunks", type="array<string>", indexing=["summary", "index"], index="enable-bm25", ), Field( name="url", type="string", indexing=["summary", "index"], index="enable-bm25", ), Field( name="binary_vectors", type="tensor<int8>(chunk{}, x[128])", indexing=["attribute", "index"], attribute=["distance-metric: hamming"], ), Field( name="int8_vectors", type="tensor<int8>(chunk{}, x[1024])", indexing=["attribute"], attribute=["paged"], ), \] ), fieldsets=\[FieldSet(name="default", fields=["chunks", "title"])\], )

We must add the schema to a Vespa [application package](https://docs.vespa.ai/en/application-packages.html). This consists of configuration files, schemas, models, and possibly even custom code (plugins).

In \[9\]:

Copied!

```
from vespa.package import ApplicationPackage

vespa_app_name = "wikipedia"
vespa_application_package = ApplicationPackage(name=vespa_app_name, schema=[my_schema])
```

from vespa.package import ApplicationPackage vespa_app_name = "wikipedia" vespa_application_package = ApplicationPackage(name=vespa_app_name, schema=[my_schema])

In the last step, we configure [ranking](https://docs.vespa.ai/en/ranking.html) by adding `rank-profile`'s to the schema.

`unpack_bits` unpacks the binary representation into a 1024-dimensional float vector [doc](https://docs.vespa.ai/en/reference/ranking-expressions.html#unpack-bits).

We define two tensor inputs, one compact binary representation that is used for the nearestNeighbor search and one full version that is used in ranking.

In \[138\]:

Copied!

```
from vespa.package import RankProfile, FirstPhaseRanking, SecondPhaseRanking, Function


rerank = RankProfile(
    name="rerank",
    inputs=[
        ("query(q_binary)", "tensor<int8>(x[128])"),
        ("query(q_int8)", "tensor<int8>(x[1024])"),
        ("query(q_full)", "tensor<float>(x[1024])"),
    ],
    functions=[
        Function(  # this returns a tensor<float>(chunk{}, x[1024]) with values -1 or 1
            name="unpack_binary_representation",
            expression="2*unpack_bits(attribute(binary_vectors)) -1",
        ),
        Function(
            name="all_chunks_cosine",
            expression="cosine_similarity(query(q_int8), attribute(int8_vectors),x)",
        ),
        Function(
            name="int8_float_dot_products",
            expression="sum(query(q_full)*unpack_binary_representation,x)",
        ),
    ],
    first_phase=FirstPhaseRanking(
        expression="reduce(int8_float_dot_products, max, chunk)"
    ),
    second_phase=SecondPhaseRanking(
        expression="reduce(all_chunks_cosine, max, chunk)"  # rescoring using the full query and a unpacked binary_vector
    ),
    match_features=[
        "distance(field, binary_vectors)",
        "all_chunks_cosine",
        "firstPhase",
        "bm25(title)",
        "bm25(chunks)",
    ],
)
my_schema.add_rank_profile(rerank)
```

from vespa.package import RankProfile, FirstPhaseRanking, SecondPhaseRanking, Function rerank = RankProfile( name="rerank", inputs=\[ ("query(q_binary)", "tensor<int8>(x[128])"), ("query(q_int8)", "tensor<int8>(x[1024])"), ("query(q_full)", "tensor<float>(x[1024])"), \], functions=\[ Function( # this returns a tensor<float>(chunk{}, x[1024]) with values -1 or 1 name="unpack_binary_representation", expression="2\*unpack_bits(attribute(binary_vectors)) -1", ), Function( name="all_chunks_cosine", expression="cosine_similarity(query(q_int8), attribute(int8_vectors),x)", ), Function( name="int8_float_dot_products", expression="sum(query(q_full)\*unpack_binary_representation,x)", ), \], first_phase=FirstPhaseRanking( expression="reduce(int8_float_dot_products, max, chunk)" ), second_phase=SecondPhaseRanking( expression="reduce(all_chunks_cosine, max, chunk)" # rescoring using the full query and a unpacked binary_vector ), match_features=[ "distance(field, binary_vectors)", "all_chunks_cosine", "firstPhase", "bm25(title)", "bm25(chunks)", ], ) my_schema.add_rank_profile(rerank)

## Deploy the application to Vespa Cloud[¶](#deploy-the-application-to-vespa-cloud)

With the configured application, we can deploy it to [Vespa Cloud](https://cloud.vespa.ai/en/).

To deploy the application to Vespa Cloud we need to create a tenant in the Vespa Cloud:

Create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/) (unless you already have one). This step requires a Google or GitHub account, and will start your [free trial](https://cloud.vespa.ai/en/free-trial).

Make note of the tenant name, it is used in the next steps.

> Note: Deployments to dev and perf expire after 7 days of inactivity, i.e., 7 days after running deploy. This applies to all plans, not only the Free Trial. Use the Vespa Console to extend the expiry period, or redeploy the application to add 7 more days.

In \[24\]:

Copied!

```
from vespa.deployment import VespaCloud
import os

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

from vespa.deployment import VespaCloud import os

# Replace with your tenant name from the Vespa Cloud Console

tenant_name = "vespa-team"

# Key is only used for CI/CD. Can be removed if logging in interactively

key = os.getenv("VESPA_TEAM_API_KEY", None) if key is not None: key = key.replace(r"\\n", "\\n") # To parse key correctly vespa_cloud = VespaCloud( tenant=tenant_name, application=vespa_app_name, key_content=key, # Key is only used for CI/CD. Can be removed if logging in interactively application_package=vespa_application_package, )

Now deploy the app to Vespa Cloud dev zone.

The first deployment typically takes 2 minutes until the endpoint is up.

In \[ \]:

Copied!

```
from vespa.application import Vespa

app: Vespa = vespa_cloud.deploy()
```

from vespa.application import Vespa app: Vespa = vespa_cloud.deploy()

## Feed the Wikipedia pages and the embedding representations[¶](#feed-the-wikipedia-pages-and-the-embedding-representations)

Read more about feeding with pyvespa in [PyVespa:reads and writes](https://vespa-engine.github.io/pyvespa/reads-writes.md).

In this case, we use a generator to yield document operations

In \[153\]:

Copied!

```
def generate_vespa_feed_documents(pages):
    for url, chunks in pages.items():
        title = None
        text_chunks = []
        binary_vectors = {}
        int8_vectors = {}
        for chunk_id, chunk in enumerate(chunks):
            title = chunk["title"]
            text = chunk["text"]
            text_chunks.append(text)
            emb_ubinary = chunk["emb_ubinary"]
            emb_ubinary = [x - 128 for x in emb_ubinary]
            emb_int8 = chunk["emb_int8"]

            binary_vectors[chunk_id] = emb_ubinary
            int8_vectors[chunk_id] = emb_int8

        vespa_json = {
            "id": url,
            "fields": {
                "doc_id": url,
                "url": url,
                "language": lang,  # Assuming `lang` is defined somewhere
                "title": title,
                "chunks": text_chunks,
                "binary_vectors": binary_vectors,
                "int8_vectors": int8_vectors,
            },
        }
        yield vespa_json
```

def generate_vespa_feed_documents(pages): for url, chunks in pages.items(): title = None text_chunks = [] binary_vectors = {} int8_vectors = {} for chunk_id, chunk in enumerate(chunks): title = chunk["title"] text = chunk["text"] text_chunks.append(text) emb_ubinary = chunk["emb_ubinary"] emb_ubinary = [x - 128 for x in emb_ubinary] emb_int8 = chunk["emb_int8"] binary_vectors[chunk_id] = emb_ubinary int8_vectors[chunk_id] = emb_int8 vespa_json = { "id": url, "fields": { "doc_id": url, "url": url, "language": lang, # Assuming `lang` is defined somewhere "title": title, "chunks": text_chunks, "binary_vectors": binary_vectors, "int8_vectors": int8_vectors, }, } yield vespa_json

In \[154\]:

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

In \[156\]:

Copied!

```
app.feed_iterable(
    iter=generate_vespa_feed_documents(pages),
    schema="page",
    callback=callback,
    max_queue_size=4000,
    max_workers=16,
    max_connections=16,
)
```

app.feed_iterable( iter=generate_vespa_feed_documents(pages), schema="page", callback=callback, max_queue_size=4000, max_workers=16, max_connections=16, )

### Querying data[¶](#querying-data)

Read more about querying Vespa in:

- [Vespa Query API](https://docs.vespa.ai/en/query-api.html)
- [Vespa Query API reference](https://docs.vespa.ai/en/reference/query-api-reference.html)
- [Vespa Query Language API (YQL)](https://docs.vespa.ai/en/query-language.html)
- [Practical Nearest Neighbor Search Guide](https://docs.vespa.ai/en/nearest-neighbor-search-guide.html)

To obtain the query embedding we use the [Cohere embed API](https://docs.cohere.com/docs/embed-api).

In \[48\]:

Copied!

```
import cohere

# Make sure that the environment variable CO_API_KEY is set to your API key
co = cohere.Client()
```

import cohere

# Make sure that the environment variable CO_API_KEY is set to your API key

co = cohere.Client()

In \[175\]:

Copied!

```
query = 'Welche britische Rockband hat das Lied "Spread Your Wings"?'
# Make sure to set input_type="search_query" when getting the embeddings for the query.
# We ask for 3 types of embeddings: float, binary, and int8
query_emb = co.embed(
    [query],
    model="embed-multilingual-v3.0",
    input_type="search_query",
    embedding_types=["float", "binary", "int8"],
)
```

query = 'Welche britische Rockband hat das Lied "Spread Your Wings"?'

# Make sure to set input_type="search_query" when getting the embeddings for the query.

# We ask for 3 types of embeddings: float, binary, and int8

query_emb = co.embed( [query], model="embed-multilingual-v3.0", input_type="search_query", embedding_types=["float", "binary", "int8"], )

Now, we use the [nearestNeighbor](https://docs.vespa.ai/en/reference/query-language-reference.html#nearestneighbor) query operator to to retrieve 1000 pages using hamming distance. This phase uses the minimum chunk-level distance for selecting pages. Essentially finding the best chunk in the page. This ensures diversity as we retrieve pages, not chunks.

These hits are exposed to the configured ranking phases that perform the re-ranking.

Notice the language parameter, for language-specific processing of the query.

In \[158\]:

Copied!

```
from vespa.io import VespaQueryResponse


response: VespaQueryResponse = app.query(
    yql="select * from page where userQuery() or ({targetHits:1000, approximate:true}nearestNeighbor(binary_vectors,q_binary))",
    ranking="rerank",
    query=query,
    language="de",  # don't guess the language of the query
    body={
        "presentation.format.tensors": "short-value",
        "input.query(q_binary)": query_emb.embeddings.binary[0],
        "input.query(q_full)": query_emb.embeddings.float[0],
        "input.query(q_int8)": query_emb.embeddings.int8[0],
    },
)
assert response.is_successful()
response.hits[0]
```

from vespa.io import VespaQueryResponse response: VespaQueryResponse = app.query( yql="select * from page where userQuery() or ({targetHits:1000, approximate:true}nearestNeighbor(binary_vectors,q_binary))", ranking="rerank", query=query, language="de", # don't guess the language of the query body={ "presentation.format.tensors": "short-value", "input.query(q_binary)": query_emb.embeddings.binary[0], "input.query(q_full)": query_emb.embeddings.float[0], "input.query(q_int8)": query_emb.embeddings.int8[0], }, ) assert response.is_successful() response.hits[0]

Out\[158\]:

```
{'id': 'id:page:page::https:/de.wikipedia.org/wiki/Spread Your Wings',
 'relevance': 0.8184863924980164,
 'source': 'wikipedia_content',
 'fields': {'matchfeatures': {'bm25(chunks)': 28.125529605038967,
   'bm25(title)': 7.345395294159827,
   'distance(field,binary_vectors)': 170.0,
   'firstPhase': 8.274434089660645,
   'all_chunks_cosine': {'0': 0.8184863924980164,
    '1': 0.6203299760818481,
    '2': 0.643619954586029,
    '3': 0.6706648468971252,
    '4': 0.524447500705719,
    '5': 0.6730406880378723}},
  'sddocname': 'page',
  'documentid': 'id:page:page::https:/de.wikipedia.org/wiki/Spread Your Wings',
  'doc_id': 'https://de.wikipedia.org/wiki/Spread%20Your%20Wings',
  'language': 'de',
  'title': 'Spread Your Wings',
  'chunks': ['Spread Your Wings ist ein Lied der britischen Rockband Queen, das von deren Bassisten John Deacon geschrieben wurde. Es ist auf dem im Oktober 1977 erschienenen Album News of the World enthalten und wurde am 10. Februar 1978 in Europa als Single mit Sheer Heart Attack als B-Seite veröffentlicht. In Nordamerika wurde es nicht als Single veröffentlicht, sondern erschien stattdessen 1980 als B-Seite des Billboard Nummer-1-Hits Crazy Little Thing Called Love. Das Lied wurde zwar kein großer Hit in den Charts, ist aber unter Queen-Fans sehr beliebt.',
   'Der Text beschreibt einen jungen Mann namens Sammy, der in einer Bar zum Putzen arbeitet (“You should’ve been sweeping/up the Emerald bar”). Während sein Chef ihn in den Strophen beschimpft und sagt, er habe keinerlei Ambitionen und solle sich mit dem zufriedengeben, was er hat (“You’ve got no real ambition,/you won’t get very far/Sammy boy don’t you know who you are/Why can’t you be happy/at the Emerald bar”), ermuntert ihn der Erzähler im Refrain, seinen Träumen nachzugehen (“spread your wings and fly away/Fly away, far away/Pull yourself together ‘cause you know you should do better/That’s because you’re a free man.”).',
   'Das Lied ist im 4/4-Takt geschrieben, beginnt in der Tonart D-Dur, wechselt in der Bridge zu deren Paralleltonart h-Moll und endet wieder mit D-Dur. Es beginnt mit einem kurzen Piano-Intro, gefolgt von der ersten Strophe, die nur mit einer akustischen Gitarre, Piano und Hi-Hats begleitet wird, und dem Refrain, in dem die E-Gitarre und das Schlagzeug hinzukommen. Die Bridge besteht aus kurzen, langsamen Gitarrentönen. Die zweite Strophe enthält im Gegensatz zur ersten beinahe von Anfang an E-Gitarren-Klänge und Schlagzeugtöne. Darauf folgt nochmals der Refrain. Das Outro ist – abgesehen von zwei kurzen Rufen – instrumental. Es besteht aus einem längeren Gitarrensolo, in dem – was für Queen äußerst ungewöhnlich ist – dieselbe Akkordfolge mehrere Male wiederholt wird und ab dem vierten Mal langsam ausblendet. Das ganze Lied enthält keinerlei Hintergrundgesang, sondern nur den Leadgesang von Freddie Mercury.',
   'Das Musikvideo wurde ebenso wie das zu We Will Rock You im Januar 1978 im Garten von Roger Taylors damaligen Anwesen Millhanger House gedreht, welches sich im Dorf Thursley im Südwesten der englischen Grafschaft Surrey befindet. Der Boden ist dabei von einer Eis- und Schneeschicht überzogen, auf der die Musiker spielten.',
   "Brian May sagte dazu später: “Looking back, it couldn't be done there – you couldn't do that!” („Wenn ich zurückschaue, hätte es nicht dort gemacht werden dürfen – man konnte das nicht tun!“)",
   'Das Lied wurde mehrfach gecovert, unter anderem von der deutschen Metal-Band Blind Guardian auf ihrem 1992 erschienenen Album Somewhere Far Beyond. Weitere Coverversionen gibt es u. a. von Jeff Scott Soto und Shawn Mars.'],
  'url': 'https://de.wikipedia.org/wiki/Spread%20Your%20Wings'}}
```

Notice the returned hits. The `relevance` is the score assigned by the second-phase expression. Also notice, that we included [bm25](https://docs.vespa.ai/en/reference/bm25.html) scores in the match-features. In this case, they do not influence ranking. The bm25 over chunks is calculated across all the elements, like if it was a bag of words or a single field string.

We now have the full Wikipedia context for all the retrieved pages. We have all the chunks and all the cosine similarity scores for all the chunks in the wikipedia page, and no need to duplicate title and url into separate retrievable units like with single-vector databases.

In RAG applications, we can now choose how much context we want to input to the generative step:

- All the chunks
- Only the best k chunks with a threshold on the cosine similarity
- The adjacent chunks of the best chunk

Or combinations of the above.

## Conclusions[¶](#conclusions)

These new Cohere binary embeddings are a huge step forward for cost-efficient vector search at scale and integrate perfectly with the rich feature set in Vespa. Including multilingual text search capabilities and hybrid search.

### Clean up[¶](#clean-up)

We can now delete the cloud instance:

In \[ \]:

Copied!

```
vespa_cloud.delete()
```

vespa_cloud.delete()
