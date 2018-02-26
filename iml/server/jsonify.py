from math import inf
from typing import List
from functools import lru_cache

import numpy as np
# import flask
from flask import jsonify
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
                'output': rule.output,
                # 'support': rule.support.tolist()
              } for rule in rl.rule_list],
        'supports': [rule.support.tolist() for rule in rl.rule_list]
        # 'discretizers': discretizer2json(rl.discretizer),
        # 'activation': rl.activation,
        # 'weights': rl.model.coefs_,
        # 'bias': rl.model.intercepts_
    }


def discretizer2json(discretizer: MDLP, data=None) -> List[dict]:
    cut_points = discretizer.cut_points_  # type: list
    category_intervals = [None] * len(cut_points)
    cut_points = [None if cut_point is None else cut_point.tolist() for cut_point in cut_points]
    maxs = discretizer.maxs_
    mins = discretizer.mins_
    # print(cut_points)
    for i, _cut_points in enumerate(cut_points):
        if _cut_points is None:
            continue
        cats = np.arange(len(_cut_points)+1)
        intervals = [[None if low == -inf else low, None if high == inf else high]
                     for low, high in discretizer.cat2intervals(cats, i)]
        category_intervals[i] = intervals
    category_ratios = [None] * len(cut_points)
    if data is not None:
        continuous = set(discretizer.continuous_features)
        for idx in range(data.shape[1]):
            col = data[:, idx]
            if idx in continuous:
                cats = discretizer.cts2cat(col, idx)
                unique_cats, _counts = np.unique(cats, return_counts=True)
                n_cats = len(discretizer.cut_points_) + 1
            else:
                unique_cats, _counts = np.unique(col, return_counts=True)
                unique_cats = unique_cats.astype(np.int)
                n_cats = int(np.max(unique_cats)) + 1
            counts = np.zeros(shape=(n_cats,))
            counts[unique_cats] = _counts
            # sorted_idx = np.argsort(unique_cats)
            category_ratios[idx] = (counts / len(col)).tolist()
        # for idx in discretizer.continuous_features:
        #     col = data[:, idx]
        #     cats = discretizer.cts2cat(col, idx)
        #     unique_cats, counts = np.unique(cats, return_counts=True)
        #     sorted_idx = np.argsort(unique_cats)
        #     category_ratios[idx] = (counts/len(col))[sorted_idx].tolist()
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
        # train_x = get_dataset(data_name, split=True, verbose=0)['train_x']
        # discretizer = discretizer2json(model.discretizer, train_x)
        # ret_dict['discretizers'] = discretizer
    elif isinstance(model, NeuralNet):
        ret_dict = nn2json(model)
    elif isinstance(model, Tree):
        ret_dict = tree2json(model)
    else:
        raise ValueError("Unsupported model of type {}".format(model.__class__))
    if isinstance(model, SurrogateMixin):
        ret_dict.update(surrogate2json(model))
    ret_dict['dataset'] = data_name
    ret_dict['name'] = model.name
    return jsonify(ret_dict)
    # ret_dict['featureNames'] = data['feature_names']
    # ret_dict['labelNames'] = data['target_names']


def data2histogram(data, n_bins: int = 20, ranges=None):
    hists = []
    for i, col in enumerate(data.T):
        counts, bin_edges = np.histogram(col, n_bins, range=None if ranges is None else ranges[i])
        bin_size = bin_edges[1] - bin_edges[0]
        bin_centers = [edge + bin_size / 2 for edge in bin_edges.tolist()[:-1]]
        hists.append({'counts': counts.tolist(), 'centers': bin_centers})
    return hists


@lru_cache(32)
def data2json(data_name, data_type='train', bins=None):
    try:
        data = get_dataset(data_name, split=True, verbose=0, discrete=True)
    except LookupError:
        return None
    if data_type == 'train' or data_type == 'test':
        x = data[data_type + '_x']
        y = data[data_type + '_y']
    # elif data_type == 'sampleTrain' or data_type == 'sampleTest':
    #     pass
    else:
        raise ValueError("Unkown data_type")
    if bins is None:
        bins = 15
        # bins = int(np.sqrt(len(y)))
        # bins = max(bins, 15)
        # print(bins)
    ranges = None if 'ranges' not in data else data['ranges']
    hists = data2histogram(x, bins, ranges)
    is_categorical = data['is_categorical']
    ret = {
        'data': x.tolist(),
        'target': y.tolist(),
        'featureNames': data['feature_names'],
        'labelNames': data['target_names'],
        'isCategorical': is_categorical.tolist() if isinstance(is_categorical, np.ndarray) else is_categorical,
        # 'continuous': [True] * x.shape[1],
        'hists': hists,
        'name': data_type,
        'ranges': data['ranges'].tolist(),
        'discretizers': discretizer2json(data['discretizer'], x)
    }

    if 'categories' in data:
        categories = data['categories']
        ret['categories'] = categories.tolist() if isinstance(categories, np.ndarray) else categories
    return jsonify(ret)
