import sys
from os import PathLike
from typing import List, Optional, Union, Mapping, Dict, IO
from pathlib import Path
from urllib.parse import urlencode

from vespa.package import (
    ModelConfig,
    Task,
    OnnxModel,
    QueryTypeField,
    Field,
    Function,
    RankProfile,
)
from vespa.json_serialization import ToJson, FromJson

#
# Optional ML dependencies
#
try:
    from torch import tensor
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        BertForSequenceClassification,
        BertTokenizerFast,
        pipeline,
    )
    from transformers.convert_graph_to_onnx import convert_pytorch
except ModuleNotFoundError:
    raise Exception("Use pip install pyvespa[ml] to install ml dependencies.")


class TextTask(Task):
    def __init__(
        self,
        model_id: str,
        model: str,
        tokenizer: Optional[str] = None,
        output_file: IO = sys.stdout,
    ):
        """
        Base class for Tasks involving text inputs.

        :param model_id: Id used to identify the model on Vespa applications.
        :param model: Id of the model as used by the model hub.
        :param tokenizer: Id of the tokenizer as used by the model hub.
        :param output_file: Output file to write output messages.
        """
        super().__init__(model_id=model_id)
        self.model = model
        self.tokenizer = tokenizer
        if not self.tokenizer:
            self.tokenizer = model
        self.output = output_file
        self._tokenizer = None
        self._model = None

    def _load_tokenizer(self):
        if not self._tokenizer:
            print("Downloading tokenizer.", file=self.output)
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

    def _create_pipeline(self):
        self._load_tokenizer()
        if not self._model:
            print("Downloading model.", file=self.output)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model)
            print("Model loaded.", file=self.output)

        _pipeline = pipeline(
            task="text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            return_all_scores=True,
            function_to_apply="None",
        )
        return _pipeline

    def export_to_onnx(self, output_path: str) -> None:
        """
        Export a model to ONNX

        :param output_path: Relative output path for the onnx model, should end in '.onnx'
        :return: None.
        """
        pipeline = self._create_pipeline()
        convert_pytorch(
            pipeline, opset=11, output=Path(output_path), use_external_format=False
        )

    def predict(self, text: str):
        """
        Predict using a local instance of the model

        :param text: text input for the task.
        :return: list with predictions
        """
        pipeline = self._create_pipeline()
        predictions = pipeline(text)[0]
        return [x["score"] for x in predictions]

    def parse_vespa_prediction(self, prediction):
        return [cell["value"] for cell in prediction["cells"]]

    def create_url_encoded_tokens(self, x):
        raise NotImplementedError


class SequenceClassification(TextTask):
    def __init__(
        self,
        model_id: str,
        model: str,
        tokenizer: Optional[str] = None,
        output_file: IO = sys.stdout,
    ):
        """
        Sequence Classification task.

        It takes a text input and returns an array of floats depending on which
        model is used to solve the task.

        :param model_id: Id used to identify the model on Vespa applications.
        :param model: Id of the model as used by the model hub. Alternatively, it can
            also be the path to the folder containing the model files, as long as
            the model config is also there.
        :param tokenizer: Id of the tokenizer as used by the model hub. Alternatively, it can
            also be the path to the folder containing the tokenizer files, as long as
            the model config is also there.
        :param output_file: Output file to write output messages.
        """
        super().__init__(
            model_id=model_id, model=model, tokenizer=tokenizer, output_file=output_file
        )

    def create_url_encoded_tokens(self, x):
        self._load_tokenizer()
        tokens = self._tokenizer(x)
        encoded_tokens = urlencode(
            {
                key: "{"
                + ",".join(
                    [
                        "{{d0: 0, d1: {}}}: {}".format(idx, x)
                        for idx, x in enumerate(value)
                    ]
                )
                + "}"
                for key, value in tokens.items()
            }
        )
        return encoded_tokens


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
            _pipeline = pipeline(
                task="text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                return_all_scores=True,
                function_to_apply="None",
            )
            convert_pytorch(_pipeline, opset=11, output=Path(output_path), use_external_format=False)
        else:
            raise ValueError("No BERT model found to be exported.")

    def onnx_model(self):
        model_file_path = self.model_id + ".onnx"
        self.export_to_onnx(output_path=model_file_path)

        return OnnxModel(
            model_name=self.model_id,
            model_file_path=model_file_path,
            inputs={
                "input_ids": "input_ids",
                "token_type_ids": "token_type_ids",
                "attention_mask": "attention_mask",
            },
            outputs={"output_0": "logits"},
        )

    def query_profile_type_fields(self):
        return [
            QueryTypeField(
                name="ranking.features.query({})".format(self.query_token_ids_name),
                type="tensor<float>(d0[{}])".format(int(self.actual_query_input_size)),
            )
        ]

    def document_fields(self, document_field_indexing):
        if not document_field_indexing:
            document_field_indexing = ["attribute", "summary"]

        return [
            Field(
                name=self.doc_token_ids_name,
                type="tensor<float>(d0[{}])".format(int(self.actual_doc_input_size)),
                indexing=document_field_indexing,
            ),
        ]

    def rank_profile(self, include_model_summary_features, **kwargs):
        constants = {"TOKEN_NONE": 0, "TOKEN_CLS": 101, "TOKEN_SEP": 102}
        if "contants" in kwargs:
            constants.update(kwargs.pop("contants"))

        functions = [
            Function(
                name="question_length",
                expression="sum(map(query({}), f(a)(a > 0)))".format(
                    self.query_token_ids_name
                ),
            ),
            Function(
                name="doc_length",
                expression="sum(map(attribute({}), f(a)(a > 0)))".format(
                    self.doc_token_ids_name
                ),
            ),
            Function(
                name="input_ids",
                expression="tokenInputIds({}, query({}), attribute({}))".format(
                    self.input_size,
                    self.query_token_ids_name,
                    self.doc_token_ids_name,
                ),
            ),
            Function(
                name="attention_mask",
                expression="tokenAttentionMask({}, query({}), attribute({}))".format(
                    self.input_size,
                    self.query_token_ids_name,
                    self.doc_token_ids_name,
                ),
            ),
            Function(
                name="token_type_ids",
                expression="tokenTypeIds({}, query({}), attribute({}))".format(
                    self.input_size,
                    self.query_token_ids_name,
                    self.doc_token_ids_name,
                ),
            ),
            Function(
                name="logit0",
                expression="onnx(" + self.model_id + ").logits{d0:0,d1:0}",
            ),
            Function(
                name="logit1",
                expression="onnx(" + self.model_id + ").logits{d0:0,d1:1}",
            ),
        ]
        if "functions" in kwargs:
            functions.extend(kwargs.pop("functions"))

        summary_features = []
        if include_model_summary_features:
            summary_features.extend(
                [
                    "logit0",
                    "logit1",
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                ]
            )
        if "summary_features" in kwargs:
            summary_features.extend(kwargs.pop("summary_features"))

        return RankProfile(
            name=self.model_id,
            constants=constants,
            functions=functions,
            summary_features=summary_features,
            **kwargs
        )

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
