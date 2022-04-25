import unittest, os

from onnxruntime import InferenceSession
from torch import Tensor
from transformers import BertForSequenceClassification
from numpy.testing import assert_almost_equal

from vespa.ml import BertModelConfig


class TestBertModelConfigTokenizerOnly(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = BertModelConfig(
            model_id="bert_tiny",
            query_input_size=4,
            doc_input_size=8,
            tokenizer=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_tokenizer"),
        )

    def test_serialization(self):
        self.assertEqual(
            self.model_config, BertModelConfig.from_dict(self.model_config.to_dict)
        )

    def test_add_model(self):
        self.assertIsNone(self.model_config.model)
        self.assertIsNone(self.model_config._model)
        self.model_config.add_model(
            model=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_model")
        )
        self.assertEqual(
            self.model_config.model,
            os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_model"),
        )
        self.assertIsInstance(self.model_config._model, BertForSequenceClassification)

    def test_doc_fields(self):
        self.assertDictEqual(
            self.model_config.doc_fields(text="this is a test"),
            {"bert_tiny_doc_token_ids": {"values": [2023, 2003, 1037, 3231, 0, 0, 0]}},
        )

    def test_query_tensor_mapping(self):
        self.assertEqual(
            self.model_config.query_tensor_mapping(text="this is a test"), [2023, 2003]
        )

    def test_create_encodings(self):
        self.assertDictEqual(
            self.model_config.create_encodings(
                queries=["this is one query", "this is another query"],
                docs=["this is one document", "this is another document"],
            ),
            {
                "input_ids": [
                    [101, 2023, 2003, 102, 2023, 2003, 2028, 6254, 102, 0, 0, 0],
                    [101, 2023, 2003, 102, 2023, 2003, 2178, 6254, 102, 0, 0, 0],
                ],
                "token_type_ids": [
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                ],
                "attention_mask": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                ],
            },
        )

    def test_create_encodings_with_tensors(self):
        encodings = self.model_config.create_encodings(
            queries=["this is one query", "this is another query"],
            docs=["this is one document", "this is another document"],
            return_tensors=True,
        )
        self.assertIsInstance(encodings["input_ids"], Tensor)
        self.assertIsInstance(encodings["token_type_ids"], Tensor)
        self.assertIsInstance(encodings["attention_mask"], Tensor)

    def test_predict(self):
        with self.assertRaises(ValueError):
            self.model_config.predict(
                queries=["this is one query", "this is another query"],
                docs=["this is one document", "this is another document"],
            )

    def test_export_to_onnx(self):
        with self.assertRaises(ValueError):
            self.model_config.export_to_onnx(output_path="test_model.onnx")


class TestBertModelConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = BertModelConfig(
            model_id="bert_tiny",
            query_input_size=4,
            doc_input_size=8,
            tokenizer=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_tokenizer"),
            model=os.path.join(os.environ["RESOURCES_DIR"], "bert_tiny_model"),
        )

    def test_serialization(self):
        self.assertEqual(
            self.model_config, BertModelConfig.from_dict(self.model_config.to_dict)
        )

    def test_predict(self):
        prediction = self.model_config.predict(
            queries=["this is one query", "this is another query"],
            docs=["this is one document", "this is another document"],
        )
        self.assertEqual(len(prediction), 2)
        self.assertEqual(len(prediction[0]), 2)
        self.assertEqual(len(prediction[1]), 2)

    @staticmethod
    def _predict_with_onnx(onnx_file_path, model_inputs):
        os.environ[
            "KMP_DUPLICATE_LIB_OK"
        ] = "True"  # required to run on mac https://stackoverflow.com/a/53014308
        m = InferenceSession(onnx_file_path)
        (out,) = m.run(input_feed=model_inputs, output_names=["output_0"])
        return out

    def test_export_to_onnx(self):
        output_path = "test_model.onnx"
        self.model_config.export_to_onnx(output_path=output_path)
        model_inputs = self.model_config.create_encodings(
            queries=["this is a query"], docs=["this is a document"]
        )
        assert_almost_equal(
            self._predict_with_onnx(output_path, model_inputs),
            self.model_config.predict(
                queries=["this is a query"], docs=["this is a document"]
            ),
        )
        os.remove(output_path)
