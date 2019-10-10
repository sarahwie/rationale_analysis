from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

import torch.distributions as D

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.attention import Attention

@Model.register("encoder_rationale_model")
class EncoderRationaleModel(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        generator: Params,
        encoder: Params,
        samples: int,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(EncoderRationaleModel, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._num_labels = self._vocabulary.get_vocab_size("labels")

        self._generator = Model.from_params(vocab=vocab, regularizer=regularizer, initializer=initializer, params=generator)
        self._encoder = Model.from_params(vocab=vocab, regularizer=regularizer, initializer=initializer, params=encoder)

        self._samples = samples

        initializer(self)

    def forward(self, document, sentence_indices, query=None, label=None, metadata=None) -> Dict[str, Any]:
        generator_dict = self._generator(document, sentence_indices, query, label, metadata)
        assert 'probs' in generator_dict

        prob_z = generator_dict['probs']
        assert len(prob_z.shape) == 2

        sampler = D.bernoulli.Bernoulli(probs=prob_z)

        loss = 0.0

        output_dict = {}

        for _ in range(self._samples) :
            sample_z, reduced_document = self._generator.reduce_document(document, sampler)
            encoder_dict = self._encoder(reduced_document, sentence_indices, query, label, metadata)

            if label is not None :
                assert 'loss' in encoder_dict

                log_prob_z = torch.log(prob_z * sample_z + (1 - prob_z) * (1 - sample_z)).sum(-1) #(B,)
                loss_sample = F.cross_entropy(encoder_dict['logits'], label, reduction='none') #(B,)

                loss += (loss_sample + loss_sample.detach() * log_prob_z).mean() / self._samples

            output_dict['probs'] = encoder_dict['probs']
            output_dict['predicted_labels'] = encoder_dict['predicted_labels']

        output_dict['loss'] = loss 
        output_dict["gold_labels"] = label
        output_dict["sentence_indices"] = sentence_indices
        output_dict["metadata"] = metadata

        self._call_metrics(output_dict)

        return output_dict

    def _decode(self, output_dict) -> Dict[str, Any]:
        new_output_dict = {}
        new_output_dict["predicted_label"] = output_dict["predicted_labels"].cpu().data.numpy()
        new_output_dict["label"] = output_dict["gold_labels"].cpu().data.numpy()
        new_output_dict["metadata"] = output_dict["metadata"]
        return new_output_dict