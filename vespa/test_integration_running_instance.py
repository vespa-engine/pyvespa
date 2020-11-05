import unittest

from random import random

from vespa.application import Vespa
from vespa.query import Union, WeakAnd, ANN, Query, RankProfile as Ranking
from vespa.evaluation import MatchRatio, Recall, ReciprocalRank


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
                embedding_model=lambda x: [random() for x in range(768)],
                hits=10,
                label="title",
            ),
        )
        rank_profile = Ranking(name="bm25", list_features=True)
        query_model = Query(match_phase=match_phase, rank_profile=rank_profile)
        #
        # Query Vespa app
        #
        query_result = app.query(
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
        #
        # Collect training data
        #
        training_data_batch = app.collect_training_data(
            labeled_data=labeled_data,
            id_field="id",
            query_model=query_model,
            number_additional_docs=2,
            fields=["rankfeatures"],
        )
        self.assertTrue(training_data_batch.shape[0] > 0)
        self.assertEqual(
            len(
                {"document_id", "query_id", "label"}.intersection(
                    set(training_data_batch.columns)
                )
            ),
            3,
        )
        #
        # Evaluate a query model
        #
        eval_metrics = [MatchRatio(), Recall(at=10), ReciprocalRank(at=10)]
        evaluation = app.evaluate(
            labeled_data=labeled_data,
            eval_metrics=eval_metrics,
            query_model=query_model,
            id_field="id",
        )
        self.assertEqual(evaluation.shape, (2, 6))
