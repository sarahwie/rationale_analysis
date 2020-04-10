from typing import Optional, Dict, Any

import torch
from transformers import AutoModel

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import F1Measure

from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel

from Rationale_Analysis.models.utils import generate_embeddings_for_pooling


@Model.register("bernoulli_bert_generator")
class BernoulliBertGenerator(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: str,
        dropout: float = 0.0,
        requires_grad: str = "none",
        pos_weight: float = 1.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(BernoulliBertGenerator, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._bert_model = AutoModel.from_pretrained(bert_model)
        self._dropout = torch.nn.Dropout(p=dropout)
        self._classification_layer = torch.nn.Linear(self._bert_model.config.hidden_size, 1, bias=False)

        if requires_grad in ["none", "all"]:
            for param in self._bert_model.parameters():
                param.requires_grad = requires_grad == "all"
        else:
            model_name_regexes = requires_grad.split(",")
            for name, param in self._bert_model.named_parameters():
                found = any([regex in name for regex in model_name_regexes])
                param.requires_grad = found

        for n, v in self._bert_model.named_parameters():
            if n.startswith("classifier"):
                v.requires_grad = True

        self._pos_weight = 1.0 / pos_weight - 1
        self._token_prf = F1Measure(1)

        initializer(self)

    def forward(self, document, query=None, label=None, metadata=None, rationale=None, **kwargs) -> Dict[str, Any]:
        #pylint: disable=arguments-differ

        bert_document = self.combine_document_query(document, query)
        
        last_hidden_states, _ = self._bert_model(
            bert_document["bert"]["wordpiece-ids"],
            attention_mask=bert_document["bert"]["wordpiece-mask"],
            position_ids=bert_document["bert"]["position-ids"],
            token_type_ids=bert_document["bert"]["type-ids"],
        )

        token_embeddings, span_mask = generate_embeddings_for_pooling(
            last_hidden_states, 
            bert_document["bert"]['document-starting-offsets'], 
            bert_document["bert"]['document-ending-offsets']
        )

        token_embeddings = util.masked_max(token_embeddings, span_mask.unsqueeze(-1), dim=2)
        token_embeddings = token_embeddings * bert_document['bert']["mask"].unsqueeze(-1)

        logits = self._classification_layer(self._dropout(token_embeddings))

        probs = torch.sigmoid(logits)[:, :, 0]
        mask = bert_document['bert']['mask']

        output_dict = {}
        output_dict["probs"] = probs * mask
        output_dict['mask'] = mask
        predicted_rationale = (probs > 0.5).long()

        output_dict["predicted_rationale"] = predicted_rationale * mask
        output_dict["prob_z"] = probs * mask

        if rationale is not None :
            rat_mask = (rationale.sum(1) > 0)
            if rat_mask.sum().long() == 0 :
                output_dict['loss'] = 0.0
            else :
                weight = torch.Tensor([1.0, self._pos_weight]).to(logits.device)
                loss = torch.nn.functional.cross_entropy(logits[rat_mask].transpose(1, 2), rationale[rat_mask], weight=weight)
                output_dict['loss'] = loss
                self._token_prf(logits[rat_mask], rationale[rat_mask], bert_document['bert']["mask"][rat_mask])


        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        try :
            metrics = self._token_prf.get_metric(reset)
        except :
            metrics = {'_p' : 0, '_r' : 0, '_f1' : 0}
            return metrics
        return dict(zip(["_p", "_r", "_f1"], metrics))
