import json
from typing import Dict
import hashlib

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, ListField, MetadataField, SpanField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

from numpy.random import RandomState


@DatasetReader.register("rationale_reader")
class RationaleReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Tokenizer,
        segment_sentences: bool = False,
        max_sequence_length: int = None,
        keep_prob: float = 1.0,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer
        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

        self._keep_prob = keep_prob

        self._bert = "bert" in token_indexers

    @overrides
    def _read(self, file_path):
        rs = RandomState(seed=1000)
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                items = json.loads(line)
                document = items["document"]
                query = items.get("query", None)
                label = items.get("label", None)
                if "annotation_id" in items:
                    annotation_id = items["annotation_id"]
                else:
                    annotation_id = hashlib.sha1(
                        document.encode("utf-8")
                        + (query.encode("utf-8") if query is not None else "".encode("utf-8"))
                    ).hexdigest()

                if label is not None:
                    label = str(label)
                if rs.random_sample() < self._keep_prob:
                    instance = self.text_to_instance(
                        annotation_id=annotation_id, document=document, query=query, label=label
                    )
                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(
        self, annotation_id: str, document: str, query: str = None, label: str = None
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}
        if self._segment_sentences:
            sentence_splits = self._sentence_segmenter.split_sentences(document)
        else:
            sentence_splits = [document]

        tokens = [Token("<S>")]
        keep_tokens = [1]
        sentence_indices = []
        for sentence in sentence_splits:
            word_tokens = self._tokenizer.tokenize(sentence)
            sentence_indices.append([len(tokens), len(tokens) + len(word_tokens)])
            tokens.extend(word_tokens)
            keep_tokens.extend([0 for _ in range(len(word_tokens))])

        if query is not None:
            if self._bert:
                query_tokens = self._tokenizer.tokenize(query)
                tokens.extend([Token('[SEP]')] + query_tokens)
                keep_tokens.extend([1 for _ in range(len(query_tokens) + 1)])
            else:
                fields["query"] = TextField(self._tokenizer.tokenize(query), self._token_indexers)

        fields["document"] = TextField(tokens, self._token_indexers)
        fields["sentence_indices"] = ListField(
            list(map(lambda x: SpanField(x[0], x[1] - 1, fields["document"]), sentence_indices))
        )

        metadata = {
            "annotation_id": annotation_id,
            "tokens": tokens,
            "keep_tokens" : keep_tokens,
            "document": document,
            "query": query,
            "convert_tokens_to_instance": self.convert_tokens_to_instance,
        }

        fields["metadata"] = MetadataField(metadata)

        if label is not None:
            fields["label"] = LabelField(label, label_namespace="labels")

        return Instance(fields)

    def convert_tokens_to_instance(self, tokens):
        fields = {}
        fields["document"] = TextField(tokens, self._token_indexers)
        return Instance(fields)
