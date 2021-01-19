from typing import Tuple, Dict, List

from sklearn.metrics import precision_recall_fscore_support
import pandas as pd


def load_data() -> Tuple[Dict, Dict]:
    """
    Helper function which loads female and male data from the `data` folder
    :return:
        Two dictionaries (female and male) where the key is the region and the value is
        a list of the most common names
    """

    female_dict, male_dict = dict(), dict()

    for dfile, g_dict in zip(
            ['data/female.txt', 'data/male.txt'],
            [female_dict, male_dict]):

        with open(dfile, 'r') as fh:

            for line in fh:
                splits = line.split()
                country = splits[0]
                g_dict[country] = set(splits[1:])

    return female_dict, male_dict


def evalute_prf(predictions: Dict, labels: List[str]):
    """
    A wrapper on sklearn's evaluator which returns the results into a nice
    `pandas` Dataframe format
    :param predictions: Object which contains the predictions of the model along with the ground truth label
    :param labels: The labels to be included in the output
    :return:
        A pandas data frame with precision, recall, F1 score per output class
    """
    y_true, y_pred = [], []

    for k, v in predictions.items():
        if v['True Label'] in labels:
            y_true.extend([v['True Label']] * len(v['Predictions']))
            y_pred.extend(v['Predictions'])

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)
    out = pd.DataFrame(
        index=labels,
        data={
            'Precision': p,
            'Recall': r,
            'F1': f1,
            'Support': s
        }
    )

    return out


def display_errors(inp_dict: Dict):
    """
    Helper function for a more complete inspection of the results.
    It expects a dict of dicts where:
        - the key of the outer dict will be used as index
        - there are three keys in the inner dict namely
             `inputs` which is the list of text items that are classified,
            `predictions` which are the predictions from the models
            and `labels` which are the ground truth labels
    :param inp_dict: Dictionary
    :return:
        A pandas data frame summarizing the accuracy and the errors per key
    """

    out = pd.DataFrame(
        index=inp_dict.keys()
    )

    for k, v in inp_dict.items():

        errors = []
        correct = 0

        label = v['True Label']

        for txt, pred in zip(v['Items'], v['Predictions']):
            if pred == label:
                correct += 1
            else:
                errors.append((txt, pred))

        out.loc[k, 'correct'] = correct
        out.loc[k, 'wrong'] = len(v['Items']) - correct
        out.loc[k, 'accuracy'] = correct / len(v['Items'])
        out.loc[k, 'True Label'] = label
        out.loc[k, 'errors (Text, Prediction)'] = ','.join([
            f'({e[0]}, {e[1]})'
            for e in errors
        ])

    return out
