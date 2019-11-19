from typing import Optional, Dict, Any

import torch
from pytorch_transformers import RobertaModel

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel

from Rationale_Analysis.models.utils import generate_embeddings_for_pooling


@Model.register("roberta_generator_model")
class RobertaGeneratorModel(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        roberta_model: str,
        dropout: float = 0.0,
        requires_grad: str = "none",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(RobertaGeneratorModel, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._roberta_model = RobertaModel.from_pretrained(roberta_model)
        self._dropout = torch.nn.Dropout(p=dropout)
        self._classification_layer = torch.nn.Linear(self._roberta_model.config.hidden_size, 2)

        self.embedding_layers = ["RobertaEmbedding"]

        if requires_grad in ["none", "all"]:
            for param in self._roberta_model.parameters():
                param.requires_grad = requires_grad == "all"
        else:
            model_name_regexes = requires_grad.split(",")
            for name, param in self._roberta_model.named_parameters():
                found = any([regex in name for regex in model_name_regexes])
                param.requires_grad = found

        for n, v in self._roberta_model.named_parameters():
            if n.startswith("classifier"):
                v.requires_grad = True

        initializer(self)

    def forward(self, document) -> Dict[str, Any]:
        input_ids = document["roberta"]
        input_mask = (input_ids != 0).long()
        starting_offsets = document["roberta-starting-offsets"]  # (B, T)

        last_hidden_states, _ = self._roberta_model(
            input_ids, attention_mask=input_mask, position_ids=document["roberta-position-ids"]
        )

        token_embeddings, span_mask = generate_embeddings_for_pooling(
            last_hidden_states, starting_offsets, document["roberta-ending-offsets"]
        )

        token_embeddings = util.masked_max(token_embeddings, span_mask.unsqueeze(-1), dim=2)
        token_embeddings = token_embeddings * document["mask"].unsqueeze(-1)

        logits = self._classification_layer(self._dropout(token_embeddings))

        assert logits.shape[0:2] == starting_offsets.shape

        probs = torch.nn.Softmax(dim=-1)(logits)[:, :, 1]
        mask = document["mask"]

        output_dict = {}
        output_dict["probs"] = probs * mask
        predicted_rationale = (probs > 0.5).long()

        output_dict["predicted_rationale"] = predicted_rationale * mask
        output_dict["prob_z"] = probs * mask

        return output_dict
