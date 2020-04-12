from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import torch.distributions as D

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel
from allennlp.data.dataset import Batch

from allennlp.training.metrics import Average

from Rationale_Analysis.models.kuma import HardKuma


@Model.register("encoder_generator_rationale_model_kuma")
class EncoderGeneratorModel(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        generator: Params,
        encoder: Params,
        samples: int,
        reg_loss_lambda: float,
        desired_length: float,
        reg_loss_mu: float = 2,
        rationale_extractor: Model = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(EncoderGeneratorModel, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._num_labels = self._vocabulary.get_vocab_size("labels")

        self._generator = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(generator)
        )
        self._encoder = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(encoder)
        )

        self._samples = samples
        self._reg_loss_lambda = reg_loss_lambda
        self._reg_loss_mu = reg_loss_mu
        self._desired_length = min(1.0, max(0.0, desired_length))
        self._rationale_extractor = rationale_extractor

        self._loss_tracks = {
            k: Average()
            for k in ["_lasso_loss", "_base_loss", "_rat_length", "_fused_lasso_loss", "_average_span_length"]
        }

        s_min = torch.Tensor([-0.1])
        s_max = torch.Tensor([1.1])
        self.support = [s_min, s_max]

        # self.lagrange_alpha = 0.5
        # self.lagrange_lr = 0.01
        # self.register_buffer('lambda0', torch.full((1,), reg_loss_lambda))
        # self.register_buffer('sparsity_ma', torch.full((1,), 0.))  # moving average

        initializer(self)

    def forward(self, document, query=None, label=None, metadata=None) -> Dict[str, Any]:
        generator_dict = self._generator(document)
        mask = util.get_text_field_mask(document)
        assert "a" in generator_dict
        assert "b" in generator_dict

        a, b = generator_dict['a'], generator_dict['b']
        a = a.clamp(1e-6, 100.)  # extreme values could result in NaNs
        b = b.clamp(1e-6, 100.)  # extreme values could result in NaNs

        output_dict = {}

        sampler = HardKuma([a, b], support=[self.support[0].to(a.device), self.support[1].to(b.device)])
        generator_dict['predicted_rationale'] = (sampler.mean() > 0.5).long() * mask
        
        if self.prediction_mode or not self.training:
            if self._rationale_extractor is None :
                sample_z = (sampler.mean() > 0.5).long() * mask
            else :
                prob_z = sampler.mean()
                sample_z = self._rationale_extractor.extract_rationale(prob_z, metadata, as_one_hot=True)
                output_dict["rationale"] = self._rationale_extractor.extract_rationale(prob_z, metadata, as_one_hot=False)
                sample_z = torch.Tensor(sample_z).to(prob_z.device).float()
        else:
            sample_z = sampler.sample()

        sample_z = sample_z * mask

        wordpiece_to_token = document['bert']['wordpiece-to-token']
        wtt0 = torch.where(wordpiece_to_token == -1, torch.tensor([0]).to(wordpiece_to_token.device), wordpiece_to_token)
        wordpiece_sample = util.batched_index_select(sample_z.unsqueeze(-1), wtt0) 
        wordpiece_sample[wordpiece_to_token.unsqueeze(-1) == -1] = 1.0

        def scale_embeddings(module, input, output) :
            output = output * wordpiece_sample
            return output

        hook = self._encoder._embedding_layer.register_forward_hook(scale_embeddings)

        encoder_dict = self._encoder(
            document=document,
            query=query,
            label=label,
            metadata=metadata,
        )

        hook.remove()

        loss = 0.0
        

        if label is not None:
            assert "loss" in encoder_dict

            base_loss = F.cross_entropy(encoder_dict["logits"], label)  # (B,)

            lasso = ((1 - sampler.pdf(0.)) * mask).sum(1)
            lengths = mask.sum(1)
            
            sparsity_loss = lasso / (lengths + 1e-9) - self._desired_length
            sparsity_loss = sparsity_loss.mean()

            self._loss_tracks["_lasso_loss"](sparsity_loss.item())

            # # moving average of the constraint
            # self.sparsity_ma = self.lagrange_alpha * self.sparsity_ma + (1 - self.lagrange_alpha) * sparsity_loss.item()

            # # update lambda
            # self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * self.sparsity_ma.detach())

            self._loss_tracks["_base_loss"](base_loss.item())
            # self._loss_tracks["_fused_lasso_loss"](self.lambda0.item())

            # loss += base_loss + min(max(self.lambda0.detach().item(), 0.01), 1.0) * sparsity_loss
            loss += base_loss + self._reg_loss_lambda * sparsity_loss

        output_dict["probs"] = encoder_dict["probs"]
        output_dict["predicted_labels"] = encoder_dict["predicted_labels"]

        output_dict["loss"] = loss
        output_dict["gold_labels"] = label
        output_dict["metadata"] = metadata

        output_dict["predicted_rationale"] = generator_dict["predicted_rationale"]

        self._loss_tracks["_rat_length"](
            util.masked_mean(generator_dict["predicted_rationale"], mask, dim=-1).mean().item()
        )

        self._call_metrics(output_dict)

        return output_dict

    def _decode(self, output_dict) -> Dict[str, Any]:
        new_output_dict = {}
        new_output_dict["predicted_label"] = output_dict["predicted_labels"].cpu().data.numpy()
        new_output_dict["label"] = output_dict["gold_labels"].cpu().data.numpy()
        new_output_dict["metadata"] = output_dict["metadata"]
        new_output_dict["rationale"] = output_dict["rationale"]
        return new_output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        base_metrics = super(EncoderGeneratorModel, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k, v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset) for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)

        reg_loss = loss_metrics["_rat_length"]
        accuracy = loss_metrics["accuracy"]

        loss_metrics["reg_accuracy"] = accuracy - 1000 * max(0, reg_loss - self._desired_length)

        return loss_metrics
