from typing import List

from torch.nn import Module
from torch import tensor
from torch.onnx import export


class BertModelConfig:
    def __init__(self, model_id, tokenizer, query_input_size, doc_input_size):
        self.model_id = model_id
        self.query_input_size = query_input_size
        self.doc_input_size = doc_input_size

        self.tokenizer = tokenizer

        self.query_token_ids_name = model_id + "_query_token_ids"
        self.actual_query_input_size = (
            query_input_size - 2
        )  # one character saved for CLS and one for SEP

        self.doc_token_ids_name = model_id + "_doc_token_ids"
        self.actual_doc_input_size = doc_input_size - 1  # one character saved for SEP

    def query_input_ids(self, queries: List[str]):
        queries_encodings = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.query_input_size - 2,
            add_special_tokens=False,
        )
        return queries_encodings["input_ids"]

    def doc_input_ids(self, docs: List[str]):
        docs_encodings = self.tokenizer(
            docs,
            truncation=True,
            max_length=self.doc_input_size - 1,
            add_special_tokens=False,
        )
        return docs_encodings["input_ids"]

    def doc_tensor(self, text):
        input_ids = self.doc_input_ids([text])[0]
        if len(input_ids) < self.actual_doc_input_size:
            input_ids = input_ids + [0] * (self.actual_doc_input_size - len(input_ids))
        return {self.doc_token_ids_name: {"values": input_ids}}

    def query_tensor_mapping(self, text) -> List[float]:
        input_ids = self.query_input_ids([text])[0]
        if len(input_ids) < self.actual_query_input_size:
            input_ids = input_ids + [0] * (self.actual_query_input_size - len(input_ids))
        return input_ids

    def create_encodings(self, queries: List[str], docs: List[str]):
        query_input_ids = self.query_input_ids(queries=queries)
        doc_input_ids = self.doc_input_ids(docs=docs)

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
            padding_length = max(128 - number_tokens, 0)
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

        encodings = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        return encodings

    def _generate_dummy_inputs(self):
        encodings = self.create_encodings(
            queries=["dummy query 1"],
            docs=["dummy document 1"],
        )
        dummy_input = {
            "input_ids": tensor(encodings["input_ids"]),
            "token_type_ids": tensor(encodings["token_type_ids"]),
            "attention_mask": tensor(encodings["attention_mask"]),
        }
        return dummy_input

    def export_to_onnx(self, model: Module, output_path: str) -> None:
        """
        Export a pytorch model to ONNX

        :param model: Model to be exported.
        :param output_path: Relative output path for the onnx model, should end in '.onnx'
        :return: None.
        """

        dummy_input = self._generate_dummy_inputs()
        input_names = ["input_ids", "token_type_ids", "attention_mask"]
        output_names = ["logits"]
        export(
            model,
            (
                dummy_input["input_ids"],
                dummy_input["token_type_ids"],
                dummy_input["attention_mask"],
            ),
            output_path,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
            opset_version=11,
        )
