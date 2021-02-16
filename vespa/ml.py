from os import PathLike
from typing import List, Optional, Union, Mapping, Dict
from pathlib import Path
from importlib.util import find_spec

from vespa.json_serialization import ToJson, FromJson

#
# Optional ML dependencies
#
try:
    from torch import tensor
    from transformers import BertForSequenceClassification, BertTokenizerFast, Pipeline
    from transformers.convert_graph_to_onnx import convert_pytorch
except ModuleNotFoundError:
    raise Exception("Use pip install pyvespa[ml] to install ml dependencies.")


class ModelConfig(object):
    def __init__(self, model_id) -> None:
        self.model_id = model_id


class BertModelConfig(ModelConfig, ToJson, FromJson["BertModelConfig"]):
    def __init__(
        self,
        model_id: str,
        query_input_size: int,
        doc_input_size: int,
        tokenizer: Union[str, PathLike],
        model: Optional[Union[str, PathLike]] = None,
    ) -> None:
        """
        BERT model configuration for Vespa applications.

        :param model_id: Unique model id to represent the model within a Vespa application.
        :param query_input_size: The size of the input vector dedicated to the query text.
        :param doc_input_size: The size of the input vector dedicated to the document text.
        :param tokenizer: The name or a path to a saved BERT model tokenizer from the transformers library.
        :param model: The name or a path to a saved model that is compatible with the `tokenizer`.
            The model is optional at construction since you might want to train it first.
            You must add a model via :func:`add_model` before deploying a Vespa application that uses this class.

        >>> bert_config = BertModelConfig(
        ...     model_id="pretrained_bert_tiny",
        ...     query_input_size=32,
        ...     doc_input_size=96,
        ...     tokenizer="google/bert_uncased_L-2_H-128_A-2",
        ... )  # doctest: +SKIP
        BertModelConfig('pretrained_bert_tiny', 32, 96, 'google/bert_uncased_L-2_H-128_A-2', None)

        >>> bert_config = BertModelConfig(
        ...     model_id="pretrained_bert_tiny",
        ...     query_input_size=32,
        ...     doc_input_size=96,
        ...     tokenizer="google/bert_uncased_L-2_H-128_A-2",
        ...     model="google/bert_uncased_L-2_H-128_A-2",
        ... )  # doctest: +SKIP
        BertModelConfig('pretrained_bert_tiny', 32, 96, 'google/bert_uncased_L-2_H-128_A-2', 'google/bert_uncased_L-2_H-128_A-2')
        """
        super().__init__(model_id=model_id)

        self.query_input_size = query_input_size
        self.doc_input_size = doc_input_size
        self.input_size = self.query_input_size + self.doc_input_size

        self.tokenizer = tokenizer
        self._tokenizer = BertTokenizerFast.from_pretrained(tokenizer)
        self._validate_tokenizer()

        self.query_token_ids_name = model_id + "_query_token_ids"
        self.actual_query_input_size = (
            query_input_size - 2
        )  # one character saved for CLS and one for SEP

        self.doc_token_ids_name = model_id + "_doc_token_ids"
        self.actual_doc_input_size = doc_input_size - 1  # one character saved for SEP

        self.model = model
        self._model = None
        if model:
            self.add_model(model=model)

    def predict(self, queries, docs) -> List:
        """
        Predict (forward pass) given queries and docs texts

        :param queries: A List of query texts.
        :param docs: A List of document texts.
        :return: A List with logits.
        """
        if not self._model:
            raise ValueError("A model needs to be added.")
        model_output = self._model(
            **self.create_encodings(queries=queries, docs=docs, return_tensors=True),
            return_dict=True
        )
        return model_output.logits.tolist()

    def _validate_tokenizer(self) -> None:
        dummy_inputs = self._generate_dummy_inputs()

        assert (
            dummy_inputs["input_ids"].shape[1]
            == dummy_inputs["token_type_ids"].shape[1]
            and dummy_inputs["token_type_ids"].shape[1]
            == dummy_inputs["attention_mask"].shape[1]
            and dummy_inputs["attention_mask"].shape[1] == self.input_size
        ), "tokenizer generates wrong input size"

    def _validate_model(self, model: BertForSequenceClassification) -> None:
        if not isinstance(model, BertForSequenceClassification):
            raise ValueError("We only support BertForSequenceClassification for now.")
        model_output = model(**self._generate_dummy_inputs(), return_dict=True)
        if len(model_output.logits.shape) != 2:
            ValueError("Model output expected to be logits vector of size 2")

    def add_model(self, model: Union[str, PathLike]) -> None:
        """
        Add a BERT model

        :param model: The name or a path to a saved model that is compatible with the `tokenizer`.
        :return: None.
        """
        _model = BertForSequenceClassification.from_pretrained(model)
        self._validate_model(model=_model)
        self.model = model
        self._model = _model

    def _query_input_ids(self, queries: List[str]):
        queries_encodings = self._tokenizer(
            queries,
            truncation=True,
            max_length=self.query_input_size - 2,
            add_special_tokens=False,
        )
        return queries_encodings["input_ids"]

    def _doc_input_ids(self, docs: List[str]):
        docs_encodings = self._tokenizer(
            docs,
            truncation=True,
            max_length=self.doc_input_size - 1,
            add_special_tokens=False,
        )
        return docs_encodings["input_ids"]

    def doc_fields(self, text) -> Dict:
        """
        Generate document fields related to the model that needs to be fed to Vespa.

        :param text: The text related to the document to be used as input to the bert model
        :return: Dict with key and values as expected by Vespa.
        """
        input_ids = self._doc_input_ids([text])[0]
        if len(input_ids) < self.actual_doc_input_size:
            input_ids = input_ids + [0] * (self.actual_doc_input_size - len(input_ids))
        return {self.doc_token_ids_name: {"values": input_ids}}

    def query_tensor_mapping(self, text) -> List[float]:
        """
        Maps query text to a tensor expected by Vespa at run time.

        :param text: Query text to be used as input to the BERT model.
        :return: Input ids expected by Vespa.
        """
        input_ids = self._query_input_ids([text])[0]
        if len(input_ids) < self.actual_query_input_size:
            input_ids = input_ids + [0] * (
                self.actual_query_input_size - len(input_ids)
            )
        return input_ids

    def create_encodings(
        self, queries: List[str], docs: List[str], return_tensors=False
    ) -> Dict:
        """
        Create BERT model encodings.

        Create BERT encodings following the same pattern used during Vespa serving. Useful to generate training data
        and ensuring training and serving compatibility.

        :param queries: A List of query texts.
        :param docs: A List of document texts.
        :return: Dict containing `input_ids`, `token_type_ids` and `attention_mask` encodings.
        """
        query_input_ids = self._query_input_ids(queries=queries)
        doc_input_ids = self._doc_input_ids(docs=docs)

        TOKEN_NONE = 0
        TOKEN_CLS = 101
        TOKEN_SEP = 102

        input_ids = []
        token_type_ids = []
        attention_mask = []
        for query_input_id, doc_input_id in zip(query_input_ids, doc_input_ids):
            # create input id
            input_id = (
                [TOKEN_CLS] + query_input_id + [TOKEN_SEP] + doc_input_id + [TOKEN_SEP]
            )
            number_tokens = len(input_id)
            padding_length = max(self.input_size - number_tokens, 0)
            input_id = input_id + [TOKEN_NONE] * padding_length
            input_ids.append(input_id)
            # create token id
            token_type_id = (
                [0] * len([TOKEN_CLS] + query_input_id + [TOKEN_SEP])
                + [1] * len(doc_input_id + [TOKEN_SEP])
                + [TOKEN_NONE] * padding_length
            )
            token_type_ids.append(token_type_id)
            # create attention_mask
            attention_mask.append([1] * number_tokens + [TOKEN_NONE] * padding_length)

        if return_tensors:
            input_ids = tensor(input_ids)
            token_type_ids = tensor(token_type_ids)
            attention_mask = tensor(attention_mask)
        encodings = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        return encodings

    def _generate_dummy_inputs(self):
        dummy_input = self.create_encodings(
            queries=["dummy query 1"], docs=["dummy document 1"], return_tensors=True
        )
        return dummy_input

    def export_to_onnx(self, output_path: str) -> None:
        """
        Export a model to ONNX

        :param output_path: Relative output path for the onnx model, should end in '.onnx'
        :return: None.
        """

        if self._model:
            pipeline = Pipeline(model=self._model, tokenizer=self._tokenizer)
            convert_pytorch(
                pipeline, opset=11, output=Path(output_path), use_external_format=False
            )
        else:
            raise ValueError("No BERT model found to be exported.")

    @staticmethod
    def from_dict(mapping: Mapping) -> "BertModelConfig":
        return BertModelConfig(
            model_id=mapping["model_id"],
            query_input_size=mapping["query_input_size"],
            doc_input_size=mapping["doc_input_size"],
            tokenizer=mapping["tokenizer"],
            model=mapping["model"],
        )

    @property
    def to_dict(self) -> Mapping:
        map = {
            "model_id": self.model_id,
            "query_input_size": self.query_input_size,
            "doc_input_size": self.doc_input_size,
            "tokenizer": self.tokenizer,
            "model": self.model,
        }
        return map

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.model_id == other.model_id
            and self.query_input_size == other.query_input_size
            and self.doc_input_size == other.doc_input_size
            and self.tokenizer == other.tokenizer
            and self.model == other.model
        )

    def __repr__(self):
        return "{0}({1}, {2}, {3}, {4}, {5})".format(
            self.__class__.__name__,
            repr(self.model_id),
            repr(self.query_input_size),
            repr(self.doc_input_size),
            repr(self.tokenizer),
            repr(self.model),
        )
