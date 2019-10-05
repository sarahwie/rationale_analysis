import json
from typing import Dict, Any
import numpy as np 

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.instance import Instance

from allennlp.data.tokenizers import Token


@DatasetReader.register("saliency_reader")
class RationaleReader(DatasetReader):
    def __init__(self, lazy: bool = False) -> None:
        super().__init__(lazy=lazy)

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                items = json.loads(line)
                saliency = np.array(items['saliency'])
                metadata = items['metadata']
                assert 'tokens' in metadata
                metadata['tokens'] = [Token(word) for word in metadata['tokens']]
                instance = self.text_to_instance(saliency=saliency, metadata=metadata)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, saliency, metadata: Dict[str, Any]) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}
        fields['attentions'] = ArrayField(saliency, padding_value=0.0)
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

