from typing import List

from vespa.package import (
    Document,
    Field,
    Schema,
    FieldSet,
    RankProfile,
    HNSW,
    ApplicationPackage,
    QueryProfile,
    QueryProfileType,
    QueryTypeField,
)
from vespa.query import QueryModel, AND, RankProfile as Ranking


class TextSearch(ApplicationPackage):
    def __init__(
        self, id_field: str, text_fields: List[str], name: str = "text_search"
    ):
        document = Document(
            fields=[
                Field(name=id_field, type="string", indexing=["attribute", "summary"])
            ]
            + [
                Field(
                    name=x,
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                )
                for x in text_fields
            ]
        )
        default_field_set = FieldSet(name="default", fields=text_fields)

        schema = Schema(
            name=name,
            document=document,
            fieldsets=[default_field_set],
            rank_profiles=[
                RankProfile(
                    name="default",
                    first_phase=" + ".join(["bm25({})".format(x) for x in text_fields]),
                ),
                RankProfile(
                    name="bm25",
                    first_phase=" + ".join(["bm25({})".format(x) for x in text_fields]),
                ),
                RankProfile(
                    name="native_rank",
                    first_phase="nativeRank({})".format(",".join(text_fields)),
                ),
            ],
        )
        super().__init__(
            name=name,
            schema=[schema],
            default_query_model=QueryModel(
                name="and_bm25", match_phase=AND(), rank_profile=Ranking(name="bm25")
            ),
        )


class QuestionAnswering(ApplicationPackage):
    def __init__(self, name: str = "qa"):
        context_document = Document(
            fields=[
                Field(
                    name="questions",
                    type="array<int>",
                    indexing=["summary", "attribute"],
                ),
                Field(name="dataset", type="string", indexing=["summary", "attribute"]),
                Field(name="context_id", type="int", indexing=["summary", "attribute"]),
                Field(
                    name="text",
                    type="string",
                    indexing=["summary", "index"],
                    index="enable-bm25",
                ),
            ]
        )
        context_schema = Schema(
            name="context",
            document=context_document,
            fieldsets=[FieldSet(name="default", fields=["text"])],
            rank_profiles=[
                RankProfile(name="bm25", inherits="default", first_phase="bm25(text)"),
                RankProfile(
                    name="nativeRank",
                    inherits="default",
                    first_phase="nativeRank(text)",
                ),
            ],
        )
        sentence_document = Document(
            inherits="context",
            fields=[
                Field(
                    name="sentence_embedding",
                    type="tensor<float>(x[512])",
                    indexing=["attribute", "index"],
                    ann=HNSW(
                        distance_metric="euclidean",
                        max_links_per_node=16,
                        neighbors_to_explore_at_insert=500,
                    ),
                )
            ],
        )
        sentence_schema = Schema(
            name="sentence",
            document=sentence_document,
            fieldsets=[FieldSet(name="default", fields=["text"])],
            rank_profiles=[
                RankProfile(
                    name="semantic-similarity",
                    inherits="default",
                    first_phase="closeness(sentence_embedding)",
                ),
                RankProfile(name="bm25", inherits="default", first_phase="bm25(text)"),
                RankProfile(
                    name="bm25-semantic-similarity",
                    inherits="default",
                    first_phase="bm25(text) + closeness(sentence_embedding)",
                ),
            ],
        )
        super().__init__(
            name=name,
            schema=[context_schema, sentence_schema],
            query_profile=QueryProfile(),
            query_profile_type=QueryProfileType(
                fields=[
                    QueryTypeField(
                        name="ranking.features.query(query_embedding)",
                        type="tensor<float>(x[512])",
                    )
                ]
            ),
        )
