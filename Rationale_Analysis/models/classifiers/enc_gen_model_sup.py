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
# from Rationale_Analysis.models.rationale_extractors.base_rationale_extractor import RationaleExtractor


@Model.register("encoder_generator_human_model")
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

        initializer(self)

    def forward(self, document, rationale=None, kept_tokens=None, query=None, label=None, metadata=None) -> Dict[str, Any]:
        generator_dict = self._generator(document, rationale)
        mask = util.get_text_field_mask(document)
        assert "probs" in generator_dict

        prob_z = generator_dict["probs"]
        assert len(prob_z.shape) == 2

        output_dict = {}

        sampler = D.bernoulli.Bernoulli(probs=prob_z)
        if self.prediction_mode or not self.training:
            if self._rationale_extractor is None :
                sample_z = generator_dict['predicted_rationale'].float()
            else :
                sample_z = self._rationale_extractor.extract_rationale(prob_z, metadata, as_one_hot=True)
                output_dict["rationale"] = self._rationale_extractor.extract_rationale(prob_z, metadata, as_one_hot=False)
                sample_z = torch.Tensor(sample_z).to(prob_z.device).float()
        else:
            sample_z = sampler.sample()

        sample_z = sample_z * mask
        reduced_document = self.regenerate_tokens(metadata, sample_z)
        encoder_dict = self._encoder(
            document=reduced_document,
            query=query,
            label=label,
            metadata=metadata,
        )

        loss = generator_dict['loss']
        

        if label is not None:
            assert "loss" in encoder_dict

            log_prob_z = sampler.log_prob(sample_z)  # (B, L)
            log_prob_z_sum = (mask * log_prob_z).sum(-1)  # (B,)
            loss_sample = F.cross_entropy(encoder_dict["logits"], label, reduction="none")  # (B,)

            sparsity = util.masked_mean(sample_z, mask, dim=-1)
            censored_lasso_loss = F.relu(sparsity - self._desired_length)

            diff = (sample_z[:, 1:] - sample_z[:, :-1]).abs()
            mask_last = mask[:, :-1]
            fused_lasso_loss = diff.sum(-1) / mask_last.sum(-1)

            self._loss_tracks["_lasso_loss"](sparsity.mean().item())
            self._loss_tracks["_fused_lasso_loss"](fused_lasso_loss.mean().item())
            self._loss_tracks["_base_loss"](loss_sample.mean().item())

            base_loss = loss_sample
            generator_loss = (
                loss_sample.detach()
                + censored_lasso_loss * self._reg_loss_lambda
                + fused_lasso_loss * (self._reg_loss_mu * self._reg_loss_lambda)
            ) * log_prob_z_sum

            loss += (base_loss + generator_loss).mean()

        output_dict["probs"] = encoder_dict["probs"]
        output_dict["predicted_labels"] = encoder_dict["predicted_labels"]

        output_dict["loss"] = loss
        output_dict["gold_labels"] = label
        output_dict["metadata"] = metadata

        output_dict["prob_z"] = generator_dict["prob_z"]
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

    def regenerate_tokens(self, metadata, sample_z):
        sample_z_cpu = sample_z.cpu().data.numpy()
        tokens = [m["tokens"] for m in metadata]
        keep_tokens = [m['keep_tokens'] for m in metadata]

        assert len(tokens) == len(sample_z_cpu)
        instances = []
        for words, mask, keep in zip(tokens, sample_z_cpu, keep_tokens):
            mask = mask[: len(words)]
            new_words = [w for i, (w, m, k) in enumerate(zip(words, mask, keep)) if i == 0 or m == 1]# or k == 1]

            instance = metadata[0]["convert_tokens_to_instance"](new_words)
            instances.append(instance)

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {k: v.to(sample_z.device) for k, v in batch["document"].items()}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        base_metrics = super(EncoderGeneratorModel, self).get_metrics(reset)
        rationale_metrics = self._generator.get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k, v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset) for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)
        loss_metrics.update(rationale_metrics)

        reg_loss = loss_metrics["_rat_length"]
        accuracy = loss_metrics["accuracy"]

        loss_metrics["reg_accuracy"] = accuracy - 1000 * max(0, reg_loss - self._desired_length)

        return loss_metrics
