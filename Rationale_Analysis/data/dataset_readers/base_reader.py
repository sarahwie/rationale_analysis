import json
from typing import Dict, List

import numpy as np
import torch
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from numpy.random import RandomState
from overrides import overrides


@DatasetReader.register("base_reader")
class BaseReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        add_rationale: bool = False,
        keep_prob: float = 1.0,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = WhitespaceTokenizer()
        self._token_indexers = token_indexers
        self._add_rationale = add_rationale
        self._keep_prob = keep_prob

        self._bert = "bert" in token_indexers

    @overrides
    def _read(self, file_path):
        rs = RandomState(seed=1000)
        with open(cached_path(file_path), "r") as data_file:
            for _, line in enumerate(data_file.readlines()):
                # if _ > 10: break
                items = json.loads(line)
                document = items["document"]
                annotation_id = items["annotation_id"]
                query = items.get("query", None)
                label = items.get("label", None)
                rationale = items.get("rationale", [])

                if label is not None:
                    label = str(label).replace(" ", "_")

                if rs.random_sample() < self._keep_prob:
                    instance = self.text_to_instance(
                        annotation_id=annotation_id, document=document, query=query, label=label, rationale=rationale
                    )
                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(
        self, annotation_id: str, document: str, query: str = None, label: str = None, rationale: List[tuple] = None
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        document_tokens = [Token(t.text, info={}) for t in self._tokenizer.tokenize(document)]
        human_rationale_labels = [0] * len(document_tokens)
        for s, e in rationale:
            for i in range(s, e):
                human_rationale_labels[i] = 1

        if query is not None:
            query_tokens = [Token("</s>", info={}), Token("<s>", info={})] + [Token(t.text, info={}) for t in self._tokenizer.tokenize(query)]
        else:
            query_tokens = []

        for index_name, indexer in self._token_indexers.items():
            if hasattr(indexer, "add_token_info"):
                indexer.add_token_info(document_tokens, index_name)
                indexer.add_token_info(query_tokens, index_name)

        fields["document"] = MetadataField({"tokens": document_tokens, "reader_object": self})
        fields["query"] = MetadataField({"tokens": query_tokens})
        fields["rationale"] = ArrayField(np.array(human_rationale_labels))
        fields["metadata"] = MetadataField(
            {
                "annotation_id": annotation_id,
                "human_rationale": rationale,
                "document": document,
                "query": query,
                "label": label,
            }
        )

        if label is not None:
            fields["label"] = LabelField(label, label_namespace="labels")

        return Instance(fields)

    def convert_tokens_to_instance(self, tokens: List[Token], query_mask: List[int]):
        fields = {}
        fields["document"] = TextField(tokens, self._token_indexers)
        fields["query_mask"] = ArrayField(np.array(query_mask).astype(int))
        return Instance(fields)

    def convert_documents_to_batch(self, documents: List[List[Token]], query_masks: List[List[int]], vocabulary) -> torch.IntTensor:
        batch = Batch([self.convert_tokens_to_instance(tokens, q_mask) for tokens, q_mask in zip(documents, query_masks)])
        batch.index_instances(vocabulary)
        batch = batch.as_tensor_dict()
        batch['document']['query_mask'] = batch['query_mask'].long()
        return batch["document"]

    def combine_document_query(self, document: List[MetadataField], query: List[MetadataField], vocabulary):
        document_tokens = [x["tokens"] + y["tokens"] for x, y in zip(document, query)]
        query_masks = [[1] * len(x["tokens"]) + [0] * len(y["tokens"]) for x, y in zip(document, query)]
        return self.convert_documents_to_batch(document_tokens, query_masks, vocabulary)
