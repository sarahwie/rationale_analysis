from Rationale_Analysis.models.rationale_extractors.base_rationale_extractor import RationaleExtractor
from allennlp.models.model import Model

import numpy as np

@Model.register("min_attention")
class MinAttentionRationaleExtractor(RationaleExtractor) :
    def __init__(self, min_attention_score: float) :
        self._min_attention_score = min_attention_score
        super().__init__()

    def forward(self, attentions, metadata) :
        rationales = self.extract_rationale(attentions=attentions, metadata=metadata)
        output_dict = {'metadata' : metadata, 'rationale' : rationales}
        return output_dict
 
    def extract_rationale(self, attentions, metadata):
        cumsumed_attention = attentions.cumsum(-1)
        sequence_attention = (
            (
                (cumsumed_attention.unsqueeze(1) - cumsumed_attention.unsqueeze(-1))
                + attentions.unsqueeze(-1)
            )
            .cpu()
            .data.numpy()
        )

        sentences = [x["tokens"] for x in metadata]
        rationales = []
        for b in range(sequence_attention.shape[0]):
            attn = sequence_attention[b]
            sentence = [x.text for x in sentences[b]]
            best_i = np.zeros((len(sentence),))
            best_j = np.zeros((len(sentence),))
            best_v = np.zeros((len(sentence),))
            for i in range(len(sentence)):
                for j in range(i, len(sentence)):
                    length = j - i
                    length_attention = attn[i, j]
                    if best_v[length] <= length_attention:
                        best_v[length] = length_attention
                        best_i[length] = i
                        best_j[length] = j

            index = min(np.searchsorted(best_v, self._min_attention_score), len(best_v) - 1)
            i, j, v = int(best_i[index]), int(best_j[index]), best_v[index]
            rationales.append([i, j + 1, v, " ".join(sentence[i : j + 1])])

        return rationales