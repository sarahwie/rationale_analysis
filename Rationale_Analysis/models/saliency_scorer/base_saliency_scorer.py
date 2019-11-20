from allennlp.models.model import Model
import torch

class SaliencyScorer(Model) :
    def __init__(self, model) :
        self._model = { 'model' : model } #This is so the model is protected from Saliency_Scorer's state_dict
        for v in self._model['model'].parameters() :
            v.requires_grad = False

        self._model['model'].prediction_mode = True
        self._model['model'].eval()

        super().__init__(self._model['model'].vocab)
        self._keepsake_param = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, **inputs) :
        output_dict = self.score(**inputs)
        return output_dict

    def decode(self, output_dict) :
        model_output_dict = self._model['model'].decode(output_dict)

        metadata = model_output_dict['metadata']
        for k in model_output_dict :
            if k != 'metadata' :
                for val, d in zip(model_output_dict[k], metadata) :
                    d[k] = val

        new_output_dict = {}
        new_output_dict.update({ 'metadata' : metadata })

        tokens_list = [m['tokens'] for m in output_dict['metadata']]

        attentions = output_dict['attentions'].cpu().data.numpy()
        new_output_dict['saliency'] = [list(m) for m in attentions]

        for i, (t, a) in enumerate(zip(tokens_list, new_output_dict['saliency'])) :
            # if len(t) != sum([x != 0 for x in a]) :
            #     breakpoint()
            new_output_dict['saliency'][i] = [round(float(x), 5) for x in a[:len(t)]]
            
        return new_output_dict
        
    def score(self, **inputs) :
        raise NotImplementedError

    def init_from_model(self) :
        pass