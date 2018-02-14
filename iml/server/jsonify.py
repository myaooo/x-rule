from math import inf
from typing import List
from functools import lru_cache

import numpy as np

from mdlp.discretization import MDLP

from iml.server import get_model, available_models, get_model_data
from iml.models import NeuralNet, SBRL, RuleSurrogate, Tree, ModelInterface, SurrogateMixin
from iml.data_processing import get_dataset


def nn2json(nn: NeuralNet) -> dict:
    return {
        'type': nn.type,
        'neurons': list(nn.neurons),
        'activation': nn.activation,
        'weights': [coef.tolist() for coef in nn.model.coefs_],
        'bias': [bias.tolist() for bias in nn.model.intercepts_]
    }


def tree2json(tree: Tree) -> dict:
    return {
        'type': 'tree',
        'root': tree.to_dict(),
        'nNodes': tree.n_nodes,
        'maxDepth': tree.max_depth,
        'nClasses': tree.n_classes,
        'nFeatures': tree.n_features,
    }


def rl2json(rl: SBRL) -> dict:
    return {
        'type': 'rule',
        'nClasses': rl.n_classes,
        'nFeatures': rl.n_features,
        'rules':
            [{
                'conditions': [{
                    'feature': feature,
                    'category': category
                } for feature, category in zip(rule.feature_indices, rule.categories)],
                'output': rule.output.tolist(),
                'support': rule.support.tolist()
              } for rule in rl.rule_list],

        # 'discretizers': discretizer2json(rl.discretizer),
        # 'activation': rl.activation,
        # 'weights': rl.model.coefs_,
        # 'bias': rl.model.intercepts_
    }


def discretizer2json(discretizer: MDLP, train_data=None) -> List[dict]:
    cut_points = discretizer.cut_points_  # type: list
    category_intervals = [None] * len(cut_points)
    cut_points = [cut_point.tolist() for cut_point in cut_points]
    maxs = discretizer.maxs_
    mins = discretizer.mins_
    # print(cut_points)
    for i, _cut_points in enumerate(cut_points):
        cats = np.arange(len(_cut_points)+1)
        intervals = [[None if low == -inf else low, None if high == inf else high]
                     for low, high in discretizer.cat2intervals(cats, i)]
        category_intervals[i] = intervals
    category_ratios = [None] * len(cut_points)
    if train_data is not None:
        for idx, col in enumerate(train_data.T):
            cats = discretizer.cts2cat(col, idx)
            unique_cats, counts = np.unique(cats, return_counts=True)
            sorted_idx = np.argsort(unique_cats)
            category_ratios[idx] = (counts/len(col))[sorted_idx].tolist()
    # print(category_intervals)
    return [{
        'cutPoints': cut_points[i],
        'intervals': category_intervals[i],
        'max': maxs[i],
        'min': mins[i],
        'ratios': category_ratios[i]
        } for i in range(len(cut_points))]


def surrogate2json(model: SurrogateMixin):
    return {
        'target': model.target.name
    }


@lru_cache(32, typed=True)
def model2json(model_name):
    # data = get_dataset(get_model_data(model))
    try:
        model = get_model(model_name)
    except FileNotFoundError:
        return None
    data_name = get_model_data(model_name)
    if isinstance(model, SBRL):
        ret_dict = rl2json(model)
        train_x = get_dataset(data_name, split=True, verbose=0)['train_x']
        discretizer = discretizer2json(model.discretizer, train_x)
        ret_dict['discretizers'] = discretizer
    elif isinstance(model, NeuralNet):
        ret_dict = nn2json(model)
    elif isinstance(model, Tree):
        ret_dict = tree2json(model)
    else:
        raise ValueError("Unsupported model of type {}".format(model.__class__))
    if isinstance(model, SurrogateMixin):
        ret_dict.update(surrogate2json(model))
    ret_dict['dataset'] = data_name

    return ret_dict
    # ret_dict['featureNames'] = data['feature_names']
    # ret_dict['labelNames'] = data['target_names']


def data2histogram(data, bins):
    hists = []
    for col in data.T:
        counts, bin_edges = np.histogram(col, bins)
        bin_size = bin_edges[1] - bin_edges[0]
        bin_centers = [edge + bin_size / 2 for edge in bin_edges.tolist()[:-1]]
        hists.append({'counts': counts.tolist(), 'centers': bin_centers})
    return hists


@lru_cache(32)
def data2json(data_name, is_train: bool, bins=None):
    try:
        data = get_dataset(data_name, split=True, verbose=0)
    except LookupError:
        return None
    x = data['train_x' if is_train else 'test_x']
    y = data['train_y' if is_train else 'test_y']
    if bins is None:
        bins = int(np.sqrt(len(y)))
        bins = max(bins, 15)
        # print(bins)
    hists = data2histogram(x, bins)
    return {
        'data': x.tolist(),
        'target': y.tolist(),
        'featureNames': data['feature_names'],
        'labelNames': data['target_names'],
        'continuous': [True] * x.shape[1],
        'hists': hists,
        'name': 'train' if is_train else 'test'
    }
