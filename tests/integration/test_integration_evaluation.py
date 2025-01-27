# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license.
# See LICENSE in the project root.

import unittest

from typing import Dict, Any

from datasets import load_dataset
from vespa.package import (
    ApplicationPackage,
    Field,
    Document,
    HNSW,
    Schema,
    RankProfile,
    Component,
    Parameter,
)
from vespa.deployment import VespaDocker
from vespa.evaluation import VespaEvaluator
from vespa.io import VespaResponse

# Reference metrics from your Sentence Transformers (semantic) runs:
# Code used to produce these metrics:

## /// script
## requires-python = ">=3.10"
## dependencies = [
##     "sentence-transformers==3.3.1",
## ]
## ///
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.evaluation import NanoBEIREvaluator

# model = SentenceTransformer('intfloat/e5-small-v2')

# datasets = ["MSMARCO"]

# evaluator = NanoBEIREvaluator(
#     dataset_names=datasets,
# )

# results = evaluator(model)
# print(results)
SENTENCE_TRANSFORMERS_REF = {
    "accuracy@1": 0.38,
    "accuracy@3": 0.64,
    "accuracy@5": 0.72,
    "accuracy@10": 0.82,
    "precision@1": 0.38,
    "precision@3": 0.21333333333333332,
    "precision@5": 0.14400000000000002,
    "precision@10": 0.08199999999999999,
    "recall@1": 0.38,
    "recall@3": 0.64,
    "recall@5": 0.72,
    "recall@10": 0.82,
    "ndcg@10": 0.6007397354752749,
    "mrr@10": 0.5308571428571428,
    "map@100": 0.5393493336728631,
}


def create_app_package() -> ApplicationPackage:
    """
    Single 'doc' schema with an 'embedding' field that uses hugging-face-embedder (E5).
    """
    return ApplicationPackage(
        name="localevaluation",
        schema=[
            Schema(
                name="doc",
                document=Document(
                    fields=[
                        Field(name="id", type="string", indexing=["summary"]),
                        Field(
                            name="text",
                            type="string",
                            indexing=["index", "summary"],
                            index="enable-bm25",
                            bolding=True,
                        ),
                        Field(
                            name="embedding",
                            type="tensor<float>(x[384])",
                            indexing=[
                                "input text",
                                "embed",  # uses default hugging-face-embedder
                                "index",
                                "attribute",
                            ],
                            ann=HNSW(distance_metric="angular"),
                            is_document_field=False,
                        ),
                    ]
                ),
                # Only one rank profile: "semantic"
                rank_profiles=[
                    RankProfile(
                        name="semantic",
                        inputs=[("query(q)", "tensor<float>(x[384])")],
                        first_phase="closeness(field, embedding)",
                    ),
                ],
            )
        ],
        components=[
            Component(
                id="e5",
                type="hugging-face-embedder",
                parameters=[
                    Parameter(
                        "transformer-model",
                        {
                            "url": "https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx"
                        },
                    ),
                    Parameter(
                        "tokenizer-model",
                        {
                            "url": "https://huggingface.co/intfloat/e5-small-v2/resolve/main/tokenizer.json"
                        },
                    ),
                ],
            ),
        ],
    )


def semantic_query_fn(query_text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Convert plain text into a JSON body for Vespa query with 'semantic' rank profile.
    """
    return {
        "yql": "select * from sources * where ({targetHits:1000}nearestNeighbor(embedding,q));",
        "query": query_text,
        "ranking": "semantic",
        "input.query(q)": f"embed({query_text})",
        "hits": top_k,
    }


class TestSemanticIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1) Build and deploy the application package
        cls.package = create_app_package()
        cls.vespa_docker = VespaDocker(port=8089)

        cls.app = cls.vespa_docker.deploy(application_package=cls.package)

        # 2) Load a portion of the NanoMSMARCO corpus
        dataset_id = "zeta-alpha-ai/NanoMSMARCO"
        corpus = load_dataset(dataset_id, "corpus", split="train", streaming=True)
        # Convert each dataset item to feed format
        vespa_feed = corpus.map(
            lambda x: {
                "id": x["_id"],
                "fields": {"text": x["text"], "id": x["_id"]},
            }
        )

        # 3) Feed the documents
        def feed_callback(response: VespaResponse, doc_id: str):
            if not response.is_successful():
                print(f"Error feeding doc {doc_id}: {response.json}")

        cls.app.feed_iterable(vespa_feed, schema="doc", callback=feed_callback)

        # 4) Prepare queries and qrels
        #    The dataset provides separate 'queries' and 'qrels' subsets
        cls.query_ds = load_dataset(dataset_id, "queries", split="train")
        cls.qrels = load_dataset(dataset_id, "qrels", split="train")

        # Convert them to dictionaries
        q_ids = cls.query_ds["_id"]
        q_texts = cls.query_ds["text"]
        cls.ids_to_query = dict(zip(q_ids, q_texts))

        rel_q_ids = cls.qrels["query-id"]
        rel_doc_ids = cls.qrels["corpus-id"]
        cls.relevant_docs = dict(zip(rel_q_ids, rel_doc_ids))

    @classmethod
    def tearDownClass(cls):
        # Clean up container
        cls.vespa_docker.container.stop(timeout=10)
        cls.vespa_docker.container.remove()

    def test_semantic_metrics_close_to_sentence_transformers(self):
        """
        Use VespaEvaluator on the 'semantic' ranking profile with the
        queries & relevant docs from the NanoMSMARCO subset,
        then compare the results to the reference values from ST.
        """

        evaluator = VespaEvaluator(
            queries=self.ids_to_query,
            relevant_docs=self.relevant_docs,
            vespa_query_fn=semantic_query_fn,
            app=self.app,
            name="semantic",
        )

        # Evaluate
        results = evaluator()
        print("Got results: ", results)

        # Compare to your reference values
        for metric, st_value in SENTENCE_TRANSFORMERS_REF.items():
            vespa_val = results[metric]
            with self.subTest(metric=metric):
                # Tolerance of 0.01
                self.assertAlmostEqual(vespa_val, st_value, delta=0.0001)


if __name__ == "__main__":
    unittest.main()
