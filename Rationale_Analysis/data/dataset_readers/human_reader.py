import json
from typing import Dict, List
import hashlib

from overrides import overrides
import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, ListField, MetadataField, SpanField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

from numpy.random import RandomState
from itertools import zip_longest, takewhile

import unicodedata

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


@DatasetReader.register("human_reader")
class HumanReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        max_sequence_length: int = None,
        human_prob: float = 1.0,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers
        self._human_prob = human_prob

        self._bert = "bert" in token_indexers

    def map_rationale_to_gold_document(self, gold_items, tokens):
        document = strip_accents(gold_items["document"].lower()).replace('\xad', '').replace(u'\u200b', '')
        rationale = gold_items["rationale"]

        assert "".join(list(takewhile(lambda x: x != "[SEP]", tokens[1:]))) == "".join(
            [x for x in document if x.strip() != ""]
        ), breakpoint()

        document_tokens = document.split()
        rationale_tokens = [0] * len(document_tokens)
        for s, e in rationale:
            for i in range(s, e):
                rationale_tokens[i] = 1

        chars = "".join(document_tokens)
        chars_ann = [r for t, r in zip(document_tokens, rationale_tokens) for _ in range(len(t))]

        tokens_rationale = [1] * len(tokens)

        start = 0
        for i in range(1, len(tokens)):
            t = tokens[i]
            if t == "[SEP]":
                break

            ann = chars_ann[start : start + len(t)]
            token_chars = "".join(chars[start : start + len(t)])
            assert t == token_chars, (t, token_chars)

            start += len(t)
            tokens_rationale[i] = 1 if np.mean(ann) > 0.5 else 0

        return tokens_rationale

    @overrides
    def _read(self, file_path):
        gold_path, predicted_path = file_path.split(";")
        rs = RandomState(seed=1000)
        with open(cached_path(gold_path), "r") as gold_file, open(cached_path(predicted_path, "r")) as predicted_file:
            for _, (gold_line, predicted_line) in enumerate(
                zip_longest(gold_file.readlines(), predicted_file.readlines())
            ):
                gold_items = json.loads(gold_line)
                predicted_items = json.loads(predicted_line)

                assert gold_items["document"] == predicted_items["metadata"]["document"], breakpoint()
                assert gold_items["annotation_id"] == predicted_items["metadata"]["annotation_id"], breakpoint()

                metadata = predicted_items["metadata"]
                tokens = metadata["tokens"]

                predicted_rationale = [x["span"] for x in predicted_items["rationale"]["spans"]]
                predicted_token_rationale = [0] * len(metadata["tokens"])
                for s, e in predicted_rationale:
                    for i in range(s, e):
                        predicted_token_rationale[i] = 1

                gold_token_rationale = self.map_rationale_to_gold_document(gold_items, tokens)

                if rs.random_sample() < self._human_prob:
                    rationale = gold_token_rationale
                else:
                    rationale = predicted_token_rationale

                instance = self.text_to_instance(
                    annotation_id=gold_items["annotation_id"],
                    document=gold_items["document"],
                    query=gold_items.get("query", None),
                    label=gold_items["label"],
                    rationale=rationale,
                    tokens_existing=tokens,
                )
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(
        self,
        annotation_id: str,
        document: str,
        query: str = None,
        label: str = None,
        rationale: List[int] = None,
        tokens_existing: List[str] = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        tokens = [Token(w) for w in tokens_existing]
        rationale_tokens = rationale

        keep_tokens = [0 if t != '[SEP]' else 1 for t in tokens]

        fields["document"] = TextField(tokens, self._token_indexers)
        fields["rationale"] = SequenceLabelField(rationale_tokens, fields["document"], "rationale_labels")

        metadata = {
            "annotation_id": annotation_id,
            "tokens": tokens,
            "keep_tokens": keep_tokens,
            "token_rationale": rationale_tokens,
            "document": document,
            "query": query,
            "convert_tokens_to_instance": self.convert_tokens_to_instance,
            "label": label,
        }

        fields["metadata"] = MetadataField(metadata)

        if label is not None:
            fields["label"] = LabelField(label, label_namespace="labels")

        return Instance(fields)

    def convert_tokens_to_instance(self, tokens):
        fields = {}
        fields["document"] = TextField(tokens, self._token_indexers)
        return Instance(fields)
