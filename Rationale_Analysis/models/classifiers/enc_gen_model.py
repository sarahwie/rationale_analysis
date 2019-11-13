from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import torch.distributions as D

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel
from allennlp.data.dataset import Batch

from allennlp.training.metrics import Average


@Model.register("encoder_generator_rationale_model")
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

        self._loss_tracks = {
            k: Average()
            for k in ["_lasso_loss", "_base_loss", "_rat_length", "_fused_lasso_loss", "_average_span_length"]
        }

        initializer(self)

    def forward(self, document, sentence_indices, query=None, label=None, metadata=None) -> Dict[str, Any]:
        generator_dict = self._generator(document)
        mask = util.get_text_field_mask(document)
        assert "probs" in generator_dict

        prob_z = generator_dict["probs"]
        assert len(prob_z.shape) == 2

        sampler = D.bernoulli.Bernoulli(probs=prob_z)

        loss = 0.0

        output_dict = {}

        for _ in range(self._samples):
            sample_z = sampler.sample() * mask
            reduced_document = self.regenerate_tokens(metadata, sample_z)
            encoder_dict = self._encoder(
                document=reduced_document,
                sentence_indices=sentence_indices,
                query=query,
                label=label,
                metadata=metadata,
            )

            if label is not None:
                assert "loss" in encoder_dict

                log_prob_z = sampler.log_prob(sample_z)  # (B, L)
                log_prob_z_sum = (mask * log_prob_z).sum(-1)  # (B,)
                loss_sample = F.cross_entropy(encoder_dict["logits"], label, reduction="none")  # (B,)

                lasso_loss = F.relu(util.masked_mean(sample_z, mask, dim=-1) - self._desired_length)
                fused_lasso_loss = util.masked_mean((sample_z[:, 1:] - sample_z[:, :-1]).abs(), mask[:, :-1], dim=-1)

                self._loss_tracks["_lasso_loss"](lasso_loss.mean().item())
                self._loss_tracks["_fused_lasso_loss"](fused_lasso_loss.mean().item())
                self._loss_tracks["_base_loss"](loss_sample.mean().item())

                base_loss = loss_sample
                generator_loss = (
                    loss_sample.detach()
                    + lasso_loss * self._reg_loss_lambda
                    + fused_lasso_loss * (self._reg_loss_mu * self._reg_loss_lambda)
                ) * log_prob_z_sum

                loss += (base_loss + generator_loss).mean() / self._samples

            output_dict["probs"] = encoder_dict["probs"]
            output_dict["predicted_labels"] = encoder_dict["predicted_labels"]

        output_dict["loss"] = loss
        output_dict["gold_labels"] = label
        output_dict["sentence_indices"] = sentence_indices
        output_dict["metadata"] = metadata

        output_dict["rationale"] = [
            [int(x) for x in np.nonzero(row)[0]] for row in ((prob_z > 0.5).long() * mask).cpu().data.numpy()
        ]
        output_dict["prob_z"] = [[float(x) for x in row] for row in prob_z.cpu().data.numpy()]
        self._loss_tracks["_rat_length"](util.masked_mean((prob_z > 0.5).long(), mask, dim=-1).mean().item())

        self._call_metrics(output_dict)

        return output_dict

    def _decode(self, output_dict) -> Dict[str, Any]:
        new_output_dict = {}
        new_output_dict["predicted_label"] = output_dict["predicted_labels"].cpu().data.numpy()
        new_output_dict["label"] = output_dict["gold_labels"].cpu().data.numpy()
        new_output_dict["metadata"] = output_dict["metadata"]
        new_output_dict["rationales"] = output_dict["rationale"]
        new_output_dict["prob_z"] = output_dict["prob_z"]
        return new_output_dict

    def regenerate_tokens(self, metadata, sample_z):
        sample_z_cpu = sample_z.cpu().data.numpy()
        tokens = [m["tokens"] for m in metadata]

        assert len(tokens) == len(sample_z_cpu)
        instances = []
        for words, mask in zip(tokens, sample_z_cpu):
            mask = mask[: len(words)]
            new_words = [w for i, (w, m) in enumerate(zip(words, mask)) if i == 0 or m == 1]

            instance = metadata[0]["convert_tokens_to_instance"](new_words)
            instances.append(instance)

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {k: v.to(sample_z.device) for k, v in batch["document"].items()}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        base_metrics = super(EncoderGeneratorModel, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k, v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset) for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)

        reg_loss = loss_metrics["_rat_length"]
        accuracy = loss_metrics["accuracy"]

        loss_metrics["reg_accuracy"] = accuracy - 1000 * max(0, reg_loss - self._desired_length)

        return loss_metrics
