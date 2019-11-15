from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.token_indexers.wordpiece_indexer import _get_token_type_ids
from allennlp.data.token_indexers import TokenIndexer

from typing import Dict, List, Callable
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token

@TokenIndexer.register("bert-pretrained-simple")
class PretrainedBertIndexerSimple(PretrainedBertIndexer) :
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        # This lowercases tokens if necessary
        text = (
            token.text.lower()
            if self._do_lowercase and token.text not in self._never_lowercase
            else token.text
            for token in tokens
        )

        # Obtain a nested sequence of wordpieces, each represented by a list of wordpiece ids
        token_wordpiece_ids = [
            [self.vocab[wordpiece] for wordpiece in self.wordpiece_tokenizer(token)]
            for token in text
        ]

        # Flattened list of wordpieces. In the end, the output of the model (e.g., BERT) should
        # have a sequence length equal to the length of this list. However, it will first be split into
        # chunks of length `self.max_pieces` so that they can be fit through the model. After packing
        # and passing through the model, it should be unpacked to represent the wordpieces in this list.
        flat_wordpiece_ids = [wordpiece for token in token_wordpiece_ids for wordpiece in token]

        # Similarly, we want to compute the token_type_ids from the flattened wordpiece ids before
        # we do the windowing; otherwise [SEP] tokens would get counted multiple times.
        flat_token_type_ids = _get_token_type_ids(flat_wordpiece_ids, self._separator_ids)

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        starting_offsets = []
        ending_offsets = []

        # If we're using initial offsets, we want to start at offset = len(text_tokens)
        # so that the first offset is the index of the first wordpiece of tokens[0].
        # Otherwise, we want to start at len(text_tokens) - 1, so that the "previous"
        # offset is the last wordpiece of "tokens[-1]".
        offset = (
            len(self._start_piece_ids)
        )

        # Count amount of wordpieces accumulated
        for token in token_wordpiece_ids:
            starting_offsets.append(offset)
            offset += len(token)
            ending_offsets.append(offset)

        wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids)]
        token_type_ids = self._extend(flat_token_type_ids)

        # Flatten the wordpiece windows
        wordpiece_ids = [wordpiece for sequence in wordpiece_windows for wordpiece in sequence]

        if len(wordpiece_ids) > self.max_pieces :
            postions_ids = [i * self.max_pieces / len(wordpiece_ids) for i in range(len(wordpiece_ids))]
        else :
            postions_ids = list(range(len(wordpiece_ids)))

        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in starting_offsets]

        return {
            index_name: wordpiece_ids,
            f"{index_name}-starting-offsets": starting_offsets,
            f"{index_name}-ending-offsets": ending_offsets,
            f"{index_name}-type-ids": token_type_ids,
            f"{index_name}-position-ids": postions_ids,
            "mask": mask,
        }