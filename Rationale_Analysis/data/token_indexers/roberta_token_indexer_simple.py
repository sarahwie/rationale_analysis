from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer

from typing import Dict, List
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token


import logging

logger = logging.getLogger(__name__)


@TokenIndexer.register("roberta-pretrained-simple")
class PretrainedRobertaIndexerSimple(PretrainedTransformerIndexer):
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary, index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary and hasattr(self.tokenizer, "vocab"):
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        # This lowercases tokens if necessary
        text = (token.text for token in tokens)

        # Obtain a nested sequence of wordpieces, each represented by a list of wordpiece ids
        token_wordpiece_ids = [
            [bpe_id for bpe_id in self.tokenizer.encode(token)] for token in text
        ]

        # Flattened list of wordpieces. In the end, the output of the model (e.g., BERT) should
        # have a sequence length equal to the length of this list. However, it will first be split into
        # chunks of length `self.max_pieces` so that they can be fit through the model. After packing
        # and passing through the model, it should be unpacked to represent the wordpieces in this list.
        flat_wordpiece_ids = [wordpiece for token in token_wordpiece_ids for wordpiece in token]

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        starting_offsets = []
        ending_offsets = []

        _start_piece_id = self.tokenizer.convert_tokens_to_ids('<s>')
        _end_piece_id = self.tokenizer.convert_tokens_to_ids('</s>')

        # If we're using initial offsets, we want to start at offset = len(text_tokens)
        # so that the first offset is the index of the first wordpiece of tokens[0].
        # Otherwise, we want to start at len(text_tokens) - 1, so that the "previous"
        # offset is the last wordpiece of "tokens[-1]".
        offset = 1

        # Count amount of wordpieces accumulated
        for token in token_wordpiece_ids:
            starting_offsets.append(offset)
            offset += len(token)
            ending_offsets.append(offset)

        wordpiece_ids = [_start_piece_id] + flat_wordpiece_ids + [_end_piece_id]

        token_type_ids = [0] * len(wordpiece_ids)

        if len(wordpiece_ids) > 512:
            postions_ids = [i * 512 / len(wordpiece_ids) for i in range(len(wordpiece_ids))]
        else:
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
