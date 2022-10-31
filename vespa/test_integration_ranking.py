import unittest
import os
import shutil
import pandas as pd
from learntorank.ranking import ListwiseRankingFramework

CONTAINER_STOP_TIMEOUT = 600


class TestListwiseRankingFrameworkDefaultValues(unittest.TestCase):
    def setUp(self) -> None:
        #
        # Load train and dev sample data
        #
        self.train_csv_file = os.path.join(
            os.environ["RESOURCES_DIR"], "beir_train_df.csv"
        )
        self.train_df = pd.read_csv(self.train_csv_file)
        self.dev_csv_file = os.path.join(os.environ["RESOURCES_DIR"], "beir_dev_df.csv")
        self.dev_df = pd.read_csv(self.dev_csv_file)

        self.ranking_framework = ListwiseRankingFramework(
            number_documents_per_query=3,
        )

    def test_dataset_from_csv_file(self):
        tf_ds = self.ranking_framework.listwise_tf_dataset_from_csv(
            file_path=os.path.join(os.environ["RESOURCES_DIR"], "beir_train_df.csv"),
            feature_names=["nativeRank", "nativeProximity"],
            shuffle_buffer_size=3,
            batch_size=1,
        )
        for features, labels in tf_ds:
            self.assertEqual(features.shape, (1, 3, 2))
            self.assertEqual(labels.shape, (1, 3))

        tf_ds = self.ranking_framework.listwise_tf_dataset_from_csv(
            file_path=os.path.join(os.environ["RESOURCES_DIR"], "beir_train_df.csv"),
            feature_names=["nativeRank", "nativeProximity"],
            shuffle_buffer_size=3,
            batch_size=2,
        )
        for features, labels in tf_ds:
            self.assertEqual(features.shape, (2, 3, 2))
            self.assertEqual(labels.shape, (2, 3))

    def test_tune_linear_model(self):

        (
            weights,
            eval_metric,
            best_hyperparams,
        ) = self.ranking_framework.fit_linear_model(
            train_data=self.train_df,
            dev_data=self.dev_df,
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

    def test_fit_linear_model(self):

        (
            weights,
            eval_metric,
            best_hyperparams,
        ) = self.ranking_framework.fit_linear_model(
            train_data=self.train_df,
            dev_data=self.dev_df,
            feature_names=[
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
            hyperparameters={"learning_rate": 0.05},
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

    def test_tune_linear_model_csv_files(self):

        (
            weights,
            eval_metric,
            best_hyperparams,
        ) = self.ranking_framework.fit_linear_model(
            train_data=self.train_csv_file,
            dev_data=self.dev_csv_file,
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
        ) = self.ranking_framework.fit_lasso_linear_model(
            train_data=self.train_df,
            dev_data=self.dev_df,
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

    def test_fit_lasso_linear_model(self):

        (
            weights,
            eval_metric,
            best_hyperparams,
        ) = self.ranking_framework.fit_lasso_linear_model(
            train_data=self.train_df,
            dev_data=self.dev_df,
            feature_names=[
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
            hyperparameters={"lambda": 0.0001, "learning_rate": 9},
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
            train_data=self.train_df,
            dev_data=self.dev_df,
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

    def test_lasso_model_search_with_fixed_hyperparams(self):

        results = self.ranking_framework.lasso_model_search(
            train_data=self.train_df,
            dev_data=self.dev_df,
            feature_names=[
                "fieldMatch(body).significance",
                "fieldMatch(body).queryCompleteness",
                "nativeRank",
            ],
            output_file=os.path.join(
                os.environ["RESOURCES_DIR"], "lasso_model_search.json"
            ),
            hyperparameter={"learning_rate": 2, "lambda": 0.001},
        )
        self.assertEqual(3, len(results))
        self.assertEqual(3, len(results[0]["weights"]["feature_names"]))
        self.assertEqual(2, len(results[1]["weights"]["feature_names"]))
        self.assertEqual(1, len(results[2]["weights"]["feature_names"]))

    def test_lasso_model_search_protected_features(self):

        results = self.ranking_framework.lasso_model_search(
            train_data=self.train_df,
            dev_data=self.dev_df,
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
            train_data=self.train_df,
            dev_data=self.dev_df,
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
            train_data=self.train_df,
            dev_data=self.dev_df,
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
            shutil.rmtree("keras_tuner")
        except FileNotFoundError:
            pass
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


class TestListwiseRankingFramework(unittest.TestCase):
    def setUp(self) -> None:
        #
        # Load train and dev sample data
        #
        self.train_df = pd.read_csv(
            os.path.join(os.environ["RESOURCES_DIR"], "beir_train_df.csv")
        )
        self.dev_df = pd.read_csv(
            os.path.join(os.environ["RESOURCES_DIR"], "beir_dev_df.csv")
        )
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
        ) = self.ranking_framework.fit_linear_model(
            train_data=self.train_df,
            dev_data=self.dev_df,
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
        ) = self.ranking_framework.fit_lasso_linear_model(
            train_data=self.train_df,
            dev_data=self.dev_df,
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
            train_data=self.train_df,
            dev_data=self.dev_df,
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
            train_data=self.train_df,
            dev_data=self.dev_df,
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
            train_data=self.train_df,
            dev_data=self.dev_df,
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
            train_data=self.train_df,
            dev_data=self.dev_df,
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
        shutil.rmtree("keras_tuner")
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
