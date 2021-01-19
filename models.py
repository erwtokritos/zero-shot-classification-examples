from typing import List, Dict
from copy import deepcopy

from transformers import pipeline
import tensorflow_hub as hub
import numpy as np


class ZeroShotModel:

    def _predict_item(self, inp, options: List[str]):
        """
        Implements the actual classification per item. Each model implements its own version
        :param inp: The text item to be classified
        :param options: The list of output classes
        :return:
            Predicts one of the given options
        """
        raise NotImplementedError('_predict_gender not implemented')

    def _predict_list(self, inputs: List[str], options: List[str]):
        """
        A simple helper function which operates on a list of inputs.
        It could be omitted but I have kept it for readability
        :param inputs: The list of items to classify
        :param options: Output classes
        :return:
        """
        return [
            self._predict_item(
                inp=inp,
                options=options
            )
            for inp in inputs
        ]

    def predict_gender(self, inputs: Dict):
        """
        It expects a list or dictionary as the input data format and returns an object of the same type
        :param inputs: Self-explanatory
        :return:
                'Female' / 'Male' decision per item
        """

        out = deepcopy(inputs)

        for k, v in inputs.items():
            out[k]['Predictions'] = self._predict_list(inputs=v['Items'], options=['Female', 'Male'])

        return out

    def predict_location(self, inputs: Dict, combinations: List[str]):
        """
        It expects a list or dictionary as the input data format and returns an object of the same type.
        For convenience in testing I am giving the regions as input
        :param inputs:
        :param combinations: The list of candidate regions for each item
        :return:
                Per item region classification
        """
        out = dict()
        regions = [c[0] for c in combinations]

        for k, v in inputs.items():
            if k not in combinations:
                continue

            out[k] = dict()
            out[k]['True Label'] = v['True Label']
            out[k]['Items'] = v['Items']
            out[k]['Predictions'] = self._predict_list(inputs=v['Items'], options=regions)
        return out


class HuggingFaceZeroShotClassifier(ZeroShotModel):

    def __init__(self):
        """
        By default it uses the RoBERTa model for zero-shot classification. For more information please refer to the
        official huggingface documentation at https://huggingface.co/transformers/main_classes/pipelines.html#transformers.ZeroShotClassificationPipeline
        or the post https://discuss.huggingface.co/t/new-pipeline-for-zero-shot-text-classification/681
        """
        self.model = pipeline("zero-shot-classification")

    def _predict_item(self, inp, options: List[str]):
        """Implementation"""

        result = self.model(inp, options)

        # by default the `labels` in the `result` structure are ordered in decreasing probability
        return result['labels'][0]


class USE4ZeroShotClassifier(ZeroShotModel):

    def __init__(self, module_url='https://tfhub.dev/google/universal-sentence-encoder/4'):
        """
        Use a `module_url` to point to a local folder if you have already downloaded the model.
        Otherwise it will try to download it the first time it is run.
        """
        self.model = hub.load(module_url)

    def _predict_item(self, inp, options: List[str]):
        """Implementation
        This is a simplistic implementation of zero-shot classification with USE. First, we embed the input and all
        available options and  we select the one with the smallest distance in terms of inner product (Cosine distance
        would return similar results as the embeddings in USE are by default normalised)
        """
        inp_repr = self.model([inp])

        # we could speed up by caching the option representations - it doesn't matter for now
        result = np.inner(inp_repr, self.model(options))

        return options[np.argmax(result)]
