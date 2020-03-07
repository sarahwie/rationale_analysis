import logging
from typing import Dict, List

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.token_indexers.token_indexer import (IndexedTokenList,
                                                        TokenIndexer)
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained-simple")
class PretrainedTransformerIndexerSimple(PretrainedTransformerIndexer):
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[int]]:
        token_wordpiece_ids = [token.info[self._index_name]["wordpiece-ids"] for token in tokens]

        flat_wordpiece_ids = [wordpiece for token in token_wordpiece_ids for wordpiece in token]

        starting_offsets = []
        ending_offsets = []

        logging.disable(logging.ERROR)
        _start_piece_id = self._tokenizer.bos_token_id or self._tokenizer.cls_token_id
        _end_piece_id = self._tokenizer.eos_token_id or self._tokenizer.sep_token_id
        logging.disable(logging.NOTSET)

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

        mask = [1 for _ in starting_offsets]
        wordpiece_mask = [1] * len(wordpiece_ids)

        return {
            "wordpiece-ids": wordpiece_ids,
            "starting-offsets": starting_offsets,
            "ending-offsets": ending_offsets,
            "type-ids": token_type_ids,
            "position-ids": postions_ids,
            "wordpiece-mask" : wordpiece_mask,
            "mask": mask,
        }

    def add_token_info(self, tokens: List[Token], index_name: str):
        self._index_name = index_name
        for token in tokens:
            wordpieces = self._tokenizer.tokenize(token.text)
            if len(wordpieces) == 0:
                token.info[index_name] = {"wordpiece-ids": [self._tokenizer.unk_token_id]}
                continue

            token.info[index_name] = {
                "wordpiece-ids": [bpe_id for bpe_id in self._tokenizer.encode(wordpieces, add_special_tokens=False)]
            }

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        # Different transformers use different padding values for tokens, but for mask and type id, the padding
        # value is always 0.
        return {
            key: torch.LongTensor(
                pad_sequence_to_length(
                    val,
                    padding_lengths[key],
                    default_value=lambda: 0
                    if "mask" in key or 'type-ids' in key
                    else self._tokenizer.pad_token_id,
                )
            )
            for key, val in tokens.items()
        }
