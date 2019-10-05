import json
from typing import Dict, List, Tuple, Union

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, ListField, MetadataField, SpanField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


@DatasetReader.register("rationale_reader")
class RationaleReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Tokenizer,
        segment_sentences: bool = False,
        max_sequence_length: int = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer
        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                items = json.loads(line)
                document = items["document"]
                query = items.get("query", None)
                label = items.get("label", None)
                if label is not None:
                    label = str(label)
                instance = self.text_to_instance(document=document, query=query, label=label)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, document: str, query: str = None, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}
        if self._segment_sentences:
            sentence_splits = self._sentence_segmenter.split_sentences(document)
        else:
            sentence_splits = [document]

        tokens = []
        sentence_indices = []
        for sentence in sentence_splits:
            word_tokens = self._tokenizer.tokenize(sentence)
            sentence_indices.append([len(tokens), len(tokens) + len(word_tokens)])
            tokens.append(word_tokens)
        fields["document"] = TextField(tokens, self._token_indexers)
        fields["sentence_indices"] = ListField(
            list(map(lambda x: SpanField(x[0], x[1] - 1, fields["document"]), sentence_indices))
        )

        if query is not None:
            fields["query"] = TextField(self._tokenizer.tokenize(query), self._token_indexers)

        if label is not None:
            fields["label"] = LabelField(label, label_namespace="labels")

        return Instance(fields)


@DatasetReader.register("bert_rationale_reader")
class BertRationaleReader(RationaleReader):
    @overrides
    def text_to_instance(self, document: str, query: str = None, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}
        if self._segment_sentences:
            sentence_splits = self._sentence_segmenter.split_sentences(document)
        else:
            sentence_splits = [document]

        tokens = []
        sentence_indices = []
        for sentence in sentence_splits:
            word_tokens = self._tokenizer.tokenize(sentence)
            sentence_indices.append([len(tokens), len(tokens) + len(word_tokens)])
            tokens.extend(word_tokens)

        if query is not None:
            tokens.append('[SEP]')
            tokens += self._tokenizer.tokenize(query)

        fields["document"] = TextField(tokens, self._token_indexers)
        fields["sentence_indices"] = ListField(
            list(map(lambda x: SpanField(x[0], x[1] - 1, fields["document"]), sentence_indices))
        )

        if label is not None:
            fields["label"] = LabelField(label, label_namespace="labels")

        metadata = {
            'tokens' : tokens,
            'document' : document,
            'query' : query
        }

        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
