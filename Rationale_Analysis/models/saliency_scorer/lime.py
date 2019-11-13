from Rationale_Analysis.models.saliency_scorer.base_saliency_scorer import SaliencyScorer
import torch
import logging
from allennlp.models.model import Model

@Model.register("simple_gradient")
class GradientSaliency(SaliencyScorer) :  
    def __init__(self, model) :
        self._embedding_layer = {}
        super().__init__(model)

        self.init_from_model()

    def init_from_model(self) :
        logging.info("Initialising from Model .... ")
        model = self._model['model']

    def score(self, document, **kwargs) :
        with torch.enable_grad() :
            

        output_dict = self._model['model'].normalize_attentions(output_dict)

        return output_dict