from Rationale_Analysis.models.rationale_extractors.base_rationale_extractor import RationaleExtractor
from allennlp.models.model import Model
import math
import numpy as np

@Model.register("max_length")
class MaxLengthRationaleExtractor(RationaleExtractor) :
    def __init__(self, max_length_ratio: float) :
        self._max_length_ratio = max_length_ratio
        super().__init__()

    def forward(self, attentions, metadata) :
        rationales = self.extract_rationale(attentions=attentions, metadata=metadata)
        output_dict = {'metadata' : metadata, 'rationale' : rationales}
        return output_dict
 
    def extract_rationale(self, attentions, metadata):
        # attentions : (B, L), metadata: List[Dict] of size B
        cumsumed_attention = attentions.cumsum(-1)

        sentences = [x["tokens"] for x in metadata]
        rationales = []
        for b in range(cumsumed_attention.shape[0]):
            attn = cumsumed_attention[b]
            sentence = [x.text for x in sentences[b]]
            best_v = np.zeros((len(sentence),))
            max_length = math.ceil(len(sentence) * self._max_length_ratio)
            for i in range(0, len(sentence) - max_length + 1):
                j = i + max_length
                best_v[i] = attn[j - 1] - (attn[i - 1] if i - 1 >= 0 else 0)
            
            index = np.argmax(best_v)
            i, j, v = index, index + max_length, best_v[index]

            top_ind = list(range(i, j))

            rationales.append({
                'document' : " ".join([x for idx, x in enumerate(sentence) if idx in top_ind]),
                'spans' : [{'span' : (i, j), 'value' : float(v)}],
                'metadata' : None
            })

        return rationales