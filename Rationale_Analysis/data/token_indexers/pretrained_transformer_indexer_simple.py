from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer

from typing import Dict, List
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token


import logging

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained-simple")
class PretrainedTransformerIndexerSimple(PretrainedTransformerIndexer):
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[int]]:
        # if not self._added_to_vocabulary and hasattr(self._tokenizer, "vocab"):
        #     self._add_encoding_to_vocabulary(vocabulary)
        #     self._added_to_vocabulary = True

        # This lowercases tokens if necessary

        token_wordpiece_ids = [token.info[self._index_name]["wordpiece_ids"] for token in tokens]

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

        # breakpoint()

        return {
            "wordpiece_ids": wordpiece_ids,
            "starting-offsets": starting_offsets,
            "ending-offsets": ending_offsets,
            "type-ids": token_type_ids,
            "position-ids": postions_ids,
            "mask": mask,
        }

    def add_token_info(self, tokens: List[Token], index_name: str):
        self._index_name = index_name
        for token in tokens:
            try:
                wordpieces = self._tokenizer.tokenize(token.text)
                if len(wordpieces) == 0:
                    token.info[index_name] = {"wordpiece_ids": [self._tokenizer.unk_token_id]}
                    continue

                token.info[index_name] = {
                    "wordpiece_ids": [bpe_id for bpe_id in self._tokenizer.encode(wordpieces, add_special_tokens=False)]
                }
            except:
                breakpoint()

