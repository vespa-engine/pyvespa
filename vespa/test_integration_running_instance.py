import unittest

from random import random
from pandas import DataFrame

from vespa.application import Vespa
from learntorank.query import (
    QueryRankingFeature,
    Union,
    WeakAnd,
    ANN,
    QueryModel,
    OR,
    Ranking,
    send_query,
    collect_vespa_features,
)
from learntorank.evaluation import MatchRatio, Recall, ReciprocalRank, evaluate


class TestRunningInstance(unittest.TestCase):
    def test_workflow(self):
        #
        # Connect to a running Vespa Application
        #
        app = Vespa(url="https://api.cord19.vespa.ai")
        #
        # Define a query model
        #
        match_phase = Union(
            WeakAnd(hits=10),
            ANN(
                doc_vector="title_embedding",
                query_vector="title_vector",
                hits=10,
                label="title",
            ),
        )
        rank_profile = Ranking(name="bm25", list_features=True)
        query_model = QueryModel(
            name="ANN_bm25",
            query_properties=[
                QueryRankingFeature(
                    name="title_vector",
                    mapping=lambda x: [random() for x in range(768)],
                )
            ],
            match_phase=match_phase,
            ranking=rank_profile,
        )
        #
        # Query Vespa app
        #
        query_result = send_query(
            app,
            query="Is remdesivir an effective treatment for COVID-19?",
            query_model=query_model,
        )
        self.assertTrue(query_result.number_documents_retrieved > 0)
        self.assertEqual(len(query_result.hits), 10)
        #
        # Define labelled data
        #
        labeled_data = [
            {
                "query_id": 0,
                "query": "Intrauterine virus infections and congenital heart disease",
                "relevant_docs": [{"id": 0, "score": 1}, {"id": 3, "score": 1}],
            },
            {
                "query_id": 1,
                "query": "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus",
                "relevant_docs": [{"id": 1, "score": 1}, {"id": 5, "score": 1}],
            },
        ]
        # equivalent data in df format
        labeled_data_df = DataFrame(
            data={
                "qid": [0, 0, 1, 1],
                "query": ["Intrauterine virus infections and congenital heart disease"]
                * 2
                + [
                    "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus"
                ]
                * 2,
                "doc_id": [0, 3, 1, 5],
                "relevance": [1, 1, 1, 1],
            }
        )

        #
        # Evaluate a query model
        #
        eval_metrics = [MatchRatio(), Recall(at=10), ReciprocalRank(at=10)]
        evaluation = evaluate(
            app=app,
            labeled_data=labeled_data,
            eval_metrics=eval_metrics,
            query_model=query_model,
            id_field="id",
        )

        self.assertEqual(evaluation.shape, (9, 1))

        #
        # AssertionError - two models with the same name
        #
        with self.assertRaises(AssertionError):
            _ = evaluate(
                app=app,
                labeled_data=labeled_data,
                eval_metrics=eval_metrics,
                query_model=[QueryModel(), QueryModel(), query_model],
                id_field="id",
            )

        evaluation = evaluate(
            app=app,
            labeled_data=labeled_data,
            eval_metrics=eval_metrics,
            query_model=[QueryModel(), query_model],
            id_field="id",
        )
        self.assertEqual(evaluation.shape, (9, 2))

        evaluation = evaluate(
            app=app,
            labeled_data=labeled_data_df,
            eval_metrics=eval_metrics,
            query_model=query_model,
            id_field="id",
            detailed_metrics=True,
        )
        self.assertEqual(evaluation.shape, (15, 1))

        evaluation = evaluate(
            app=app,
            labeled_data=labeled_data_df,
            eval_metrics=eval_metrics,
            query_model=query_model,
            id_field="id",
            detailed_metrics=True,
            per_query=True,
        )
        self.assertEqual(evaluation.shape, (2, 7))

    def test_collect_training_data(self):
        app = Vespa(url="https://api.cord19.vespa.ai")
        query_model = QueryModel(
            match_phase=OR(), ranking=Ranking(name="bm25", list_features=True)
        )
        labeled_data = [
            {
                "query_id": 0,
                "query": "Intrauterine virus infections and congenital heart disease",
                "relevant_docs": [{"id": 0, "score": 1}, {"id": 3, "score": 1}],
            },
            {
                "query_id": 1,
                "query": "Clinical and immunologic studies in identical twins discordant for systemic lupus erythematosus",
                "relevant_docs": [{"id": 1, "score": 1}, {"id": 5, "score": 1}],
            },
        ]
        rank_features = collect_vespa_features(
            app=app,
            labeled_data=labeled_data,
            id_field="id",
            query_model=query_model,
            number_additional_docs=2,
            fields=["rankfeatures"],
        )
        self.assertTrue(rank_features.shape[0] > 4)
        # It should have at least one rank feature in addition to document_id, query_id and	label
        self.assertTrue(rank_features.shape[1] > 3)
        rank_features = collect_vespa_features(
            app=app,
            labeled_data=labeled_data,
            id_field="id",
            query_model=query_model,
            number_additional_docs=2,
            fields=["rankfeatures"],
            keep_features=["textSimilarity(title).score"],
        )
        self.assertTrue(rank_features.shape[0] > 4)
        # It should have at least one rank feature in addition to document_id, query_id and	label
        self.assertTrue(rank_features.shape[1] == 4)
