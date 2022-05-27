import unittest
import tempfile
import os
import json
import pandas as pd
from vespa.deployment import VespaDocker
from vespa.query import QueryModel, WeakAnd, RankProfile as Ranking
from vespa.experimental.ranking import (
    BeirData,
    SparseBeirApplicationPackage,
    SparseBeirApp,
    ListwiseRankingFramework,
)


class TestBeirData(unittest.TestCase):
    def setUp(self) -> None:
        self.data_source = BeirData(data_dir=os.environ["RESOURCES_DIR"])

    def test_sample_data(self):
        #
        # Load the full dataset
        #
        full_data = self.data_source.load_data(file_name="beir_data.json")
        #
        # Sample from dataset
        #
        sample_data = self.data_source.sample_data(
            data=full_data,
            number_positive_samples=2,
            number_negative_samples=5,
            split_types=["train", "dev"],
        )
        self.assertEqual(len(sample_data["corpus"]), 9)
        self.assertLessEqual(sample_data["corpus"].items(), full_data["corpus"].items())
        self.assertEqual(len(sample_data["split"]["train"]["qrels"]), 2)
        self.assertLessEqual(
            sample_data["split"]["train"]["qrels"].items(),
            full_data["split"]["train"]["qrels"].items(),
        )
        self.assertEqual(len(sample_data["split"]["train"]["queries"]), 2)
        self.assertLessEqual(
            sample_data["split"]["train"]["queries"].items(),
            full_data["split"]["train"]["queries"].items(),
        )
        self.assertEqual(len(sample_data["split"]["dev"]["qrels"]), 2)
        self.assertLessEqual(
            sample_data["split"]["dev"]["qrels"].items(),
            full_data["split"]["dev"]["qrels"].items(),
        )
        self.assertEqual(len(sample_data["split"]["dev"]["queries"]), 2)
        self.assertLessEqual(
            sample_data["split"]["dev"]["queries"].items(),
            full_data["split"]["dev"]["queries"].items(),
        )
        #
        # Save sample data
        #
        self.data_source.save_data(sample_data, file_name="sample.json")
        #
        # Load sample data
        #
        loaded_sample = self.data_source.load_data(file_name="sample.json")
        self.assertDictEqual(sample_data, loaded_sample)

    def tearDown(self) -> None:
        os.remove(os.path.join(os.environ["RESOURCES_DIR"], "sample.json"))


class TestSparseBeirApp(unittest.TestCase):
    def setUp(self) -> None:
        with open(
            os.path.join(os.environ["RESOURCES_DIR"], "beir_data_sample.json"), "r"
        ) as f:
            self.data_sample = json.load(f)

    def test_end_to_end_workflow(self):

        # create application package
        app_package = SparseBeirApplicationPackage()

        # # Add linear model ranking function
        app_package.add_first_phase_linear_model(
            name="first_phase_test",
            weights={"bm25(body)": 0.2456, "nativeRank(body)": -0.743},
        )
        self.assertEqual(
            "bm25(body) * 0.2456 + nativeRank(body) * -0.743",
            app_package.schema.rank_profiles["first_phase_test"].first_phase,
        )

        # deploy application package
        vespa_docker = VespaDocker(port=8089)
        app = vespa_docker.deploy(app_package)

        # interact with app
        app = SparseBeirApp(app)

        # feed app
        feed_responses = app.feed(self.data_sample)
        self.assertEqual(
            9, sum([1 if x.status_code == 200 else 0 for x in feed_responses])
        )

        # collect data
        number_additional_docs = 2
        train_df = app.collect_vespa_features(
            data=self.data_sample,
            split_type="train",
            number_additional_docs=number_additional_docs,
        )
        dev_df = app.collect_vespa_features(
            data=self.data_sample,
            split_type="dev",
            number_additional_docs=number_additional_docs,
        )
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(dev_df, pd.DataFrame)

        # evaluate ranking function
        query_model = QueryModel(
            name="first_phase_test",
            match_phase=WeakAnd(hits=number_additional_docs),
            rank_profile=Ranking(name="first_phase_test"),
        )
        evaluation = app.evaluate(self.data_sample, query_model=query_model)
        self.assertGreaterEqual(evaluation, 0)


class TestListwiseRankingFramework(unittest.TestCase):
    def setUp(self) -> None:
        #
        # Load train and dev sample data
        #
        self.train_df = pd.read_csv("resources/beir_train_df.csv")
        self.dev_df = pd.read_csv("resources/beir_dev_df.csv")
        #
        # Data config
        #
        number_documents_per_query = 3
        batch_size = 3
        #
        # Hyperparameter exploration
        #
        tuner_max_trials = 2
        tuner_executions_per_trial = 1
        tuner_epochs = 10
        tuner_early_stop_patience = 5
        #
        # Final run with best hyperparameters
        #
        final_epochs = 10
        #
        # NDCG metric parameters
        #
        top_n = 10
        #
        # Lasso search parameters
        #
        l1_penalty_range = [1e-4, 1e-2]
        learning_rate_range = [1e-2, 1e2]

        self.ranking_framework = ListwiseRankingFramework(
            number_documents_per_query=number_documents_per_query,
            batch_size=batch_size,
            tuner_max_trials=tuner_max_trials,
            tuner_executions_per_trial=tuner_executions_per_trial,
            tuner_epochs=tuner_epochs,
            tuner_early_stop_patience=tuner_early_stop_patience,
            final_epochs=final_epochs,
            top_n=top_n,
            l1_penalty_range=l1_penalty_range,
            learning_rate_range=learning_rate_range,
        )

    def test_tune_linear_model(self):

        (
            weights,
            eval_metric,
            best_hyperparams,
        ) = self.ranking_framework.tune_linear_model(
            train_df=self.train_df,
            dev_df=self.dev_df,
            feature_names=[
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
        )
        #
        # Check weights format
        #
        self.assertEqual(
            [
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
            weights["feature_names"],
        )
        self.assertEqual(3, len(weights["linear_model_weights"]))
        #
        # Check evaluation metric
        #
        self.assertGreater(eval_metric, 0)
        #
        # Check best hyperparams
        #
        self.assertEqual(list(best_hyperparams.keys()), ["learning_rate"])

    def test_tune_lasso_linear_model(self):

        (
            weights,
            eval_metric,
            best_hyperparams,
        ) = self.ranking_framework.tune_lasso_linear_model(
            train_df=self.train_df,
            dev_df=self.dev_df,
            feature_names=[
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
        )
        #
        # Check weights format
        #
        self.assertEqual(
            [
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
            weights["feature_names"],
        )
        self.assertEqual(3, len(weights["normalization_mean"]))
        self.assertEqual(3, len(weights["normalization_sd"]))
        self.assertEqual(6, weights["normalization_number_data"])
        self.assertEqual(3, len(weights["linear_model_weights"]))
        #
        # Check evaluation metric
        #
        self.assertGreater(eval_metric, 0)
        #
        # Check best hyperparams
        #
        self.assertEqual(list(best_hyperparams.keys()), ["lambda", "learning_rate"])

    def test_lasso_model_search(self):

        results = self.ranking_framework.lasso_model_search(
            train_df=self.train_df,
            dev_df=self.dev_df,
            feature_names=[
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
            output_file=os.path.join(
                os.environ["RESOURCES_DIR"], "lasso_model_search.json"
            ),
        )
        self.assertEqual(3, len(results))
        self.assertEqual(3, len(results[0]["weights"]["feature_names"]))
        self.assertEqual(2, len(results[1]["weights"]["feature_names"]))
        self.assertEqual(1, len(results[2]["weights"]["feature_names"]))

    def test_lasso_model_search_protected_features(self):

        results = self.ranking_framework.lasso_model_search(
            train_df=self.train_df,
            dev_df=self.dev_df,
            feature_names=[
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
            protected_features=["fieldMatch(body).queryCompleteness", "nativeRank"],
            output_file=os.path.join(
                os.environ["RESOURCES_DIR"], "lasso_model_search.json"
            ),
        )
        self.assertEqual(2, len(results))
        self.assertEqual(3, len(results[0]["weights"]["feature_names"]))
        self.assertEqual(2, len(results[1]["weights"]["feature_names"]))
        self.assertEqual(
            ["fieldMatch(body).queryCompleteness", "nativeRank"],
            results[1]["weights"]["feature_names"],
        )

    def test_forward_selection_model_search(self):

        results = self.ranking_framework.forward_selection_model_search(
            train_df=self.train_df,
            dev_df=self.dev_df,
            feature_names=[
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
            output_file=os.path.join(
                os.environ["RESOURCES_DIR"], "forward_selection_model_search.json"
            ),
        )
        self.assertEqual(6, len(results))
        self.assertEqual(1, len(results[0]["weights"]["feature_names"]))
        self.assertEqual(1, len(results[1]["weights"]["feature_names"]))
        self.assertEqual(1, len(results[2]["weights"]["feature_names"]))
        self.assertEqual(2, len(results[3]["weights"]["feature_names"]))
        self.assertEqual(2, len(results[4]["weights"]["feature_names"]))
        self.assertEqual(3, len(results[5]["weights"]["feature_names"]))

    def test_forward_selection_model_search_with_protected_features(self):

        results = self.ranking_framework.forward_selection_model_search(
            train_df=self.train_df,
            dev_df=self.dev_df,
            feature_names=[
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
            protected_features=["fieldMatch(body).significance"],
            output_file=os.path.join(
                os.environ["RESOURCES_DIR"], "forward_selection_model_search.json"
            ),
        )
        self.assertEqual(4, len(results))
        self.assertEqual(1, len(results[0]["weights"]["feature_names"]))
        self.assertEqual(2, len(results[1]["weights"]["feature_names"]))
        self.assertEqual(2, len(results[2]["weights"]["feature_names"]))
        self.assertEqual(3, len(results[3]["weights"]["feature_names"]))

    def tearDown(self) -> None:
        try:
            os.remove(
                os.path.join(os.environ["RESOURCES_DIR"], "lasso_model_search.json")
            )
        except OSError:
            pass
        try:
            os.remove(
                os.path.join(
                    os.environ["RESOURCES_DIR"], "forward_selection_model_search.json"
                )
            )
        except OSError:
            pass
