from typing import Optional, Dict, Any

import torch
from pytorch_transformers import BertForSequenceClassification

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel


@Model.register("bert_rationale_model")
class BertRationaleModel(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: str,
        dropout: float = 0.0,
        requires_grad: str = "none",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(BertRationaleModel, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._num_labels = self._vocabulary.get_vocab_size("labels")
        self._bert_model = BertForSequenceClassification.from_pretrained(
            bert_model, num_labels=self._num_labels, hidden_dropout_prob=dropout, output_attentions=True
        )

        self.embedding_layers = ["BertEmbedding"]

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

        initializer(self)

    def forward(self, document, sentence_indices, query=None, label=None, metadata=None) -> Dict[str, Any]:
        input_ids = document["bert"]
        input_mask = (input_ids != 0).long()

        outputs = self._bert_model(
            input_ids, attention_mask=input_mask, position_ids=document["bert-position-ids"], labels=label
        )

        loss = 0
        if label is not None :
            loss, logits, attentions = outputs
        else :
            logits, attentions = outputs

        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}
        attentions = attentions[-1][:, :, 0, :].mean(1)

        output_dict["logits"] = logits
        output_dict["loss"] = loss
        output_dict["probs"] = probs
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["gold_labels"] = label
        output_dict["sentence_indices"] = sentence_indices
        output_dict["attentions"] = attentions
        output_dict["metadata"] = metadata

        output_dict["input_ids"] = input_ids
        output_dict["input_mask"] = input_mask
        output_dict["input_starting_offsets"] = document["bert-starting-offsets"]
        output_dict["input_ending_offsets"] = document["bert-ending-offsets"]

        if label is not None :
            self._call_metrics(output_dict)

        return output_dict

    def _decode(self, output_dict) -> Dict[str, Any]:
        new_output_dict = {}
        new_output_dict["predicted_label"] = output_dict["predicted_labels"].cpu().data.numpy()
        new_output_dict["label"] = output_dict["gold_labels"].cpu().data.numpy()
        new_output_dict["metadata"] = output_dict["metadata"]
        return new_output_dict

    def normalize_attentions(self, output_dict) -> None:
        attentions, input_offsets, input_mask = (
            output_dict["attentions"],
            output_dict["input_starting_offsets"],
            output_dict["input_mask"],
        )
        cumsumed_attentions = attentions.cumsum(-1)

        starting_offsets = input_offsets
        starting_offsets = torch.cat(
            [starting_offsets, torch.zeros((starting_offsets.shape[0], 1)).long().to(starting_offsets.device)], dim=-1
        )
        starting_offsets += (starting_offsets == 0) * (input_mask.sum(-1)[:, None] - 1)

        ending_offsets = starting_offsets[:, 1:]
        starting_offsets = starting_offsets[:, :-1]
        ending_offsets = ending_offsets - 1
        starting_offsets = starting_offsets - 1

        cumsumed_attentions = cumsumed_attentions.unsqueeze(-1)
        output_dict["attentions"] = (
            util.batched_index_select(cumsumed_attentions, ending_offsets)
            - util.batched_index_select(cumsumed_attentions, starting_offsets)
        ).squeeze(-1) * (input_offsets != 0).float()

        output_dict["attentions"] = output_dict["attentions"] / output_dict["attentions"].sum(-1, keepdim=True)

        return output_dict
