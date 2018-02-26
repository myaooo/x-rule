from typing import Union, List
from functools import lru_cache

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score
from flask import jsonify

from iml.models import RuleList, Tree, ModelBase
from iml.data_processing import get_dataset
from iml.server.model_cache import get_model, get_model_data


@lru_cache(32)
def model_metric(model_name, data):
    try:
        model = get_model(model_name)
    except FileNotFoundError:
        return None

    if data == 'train' or data == 'test':
        dataset = get_dataset(get_model_data(model_name), split=True)
        if data == 'train':
            x = dataset['train_x']
            y = dataset['train_y']
        else:
            x = dataset['test_x']
            y = dataset['test_y']
    # elif data == 'sample_train' or 'sample_test':
    #     pass
    else:
        raise ValueError("Unknown data {}".format(data))
    conf_mat = confusion_matrix(y, model.predict(x))
    y_pred = model.predict_prob(x)
    # if y_pred.shape[1] == 2:
    #     auc = roc_auc_score(y, y_pred[:, 1])
    # else:
    auc = roc_auc_score(label2binary(y), y_pred, average=None)
    ret = {
        'confusionMatrix': conf_mat,
        'auc': auc
    }
    return jsonify(ret)


@lru_cache(32)
def get_support(model_name, data_type):
    model = get_model(model_name)
    if data_type == 'train' or data_type == 'test':
        dataset = get_dataset(get_model_data(model_name), split=True)
        x = dataset[data_type + '_x']
        y = dataset[data_type + '_y']
    # elif data_type == 'sample_train' or 'sample_test':
    #     pass
    else:
        raise ValueError('Unknown data type {}. Should be one of [train, test, sample_train, sample_test]'.format(data_type))
    supports = compute_support(model, x, y)
    return jsonify(supports)


def compute_support(model: ModelBase, x: np.ndarray, y: np.ndarray):
    if isinstance(model, RuleList):
        # Return a matrix of shape (n_rules, n_classes)
        return model.compute_support(x, y, transform=True)
    if isinstance(model, Tree):
        return model.compute_support(x, y, transform=True)
    return None


def label2binary(y):
    return OneHotEncoder().fit_transform(y.reshape([-1, 1])).toarray()


@lru_cache(32)
def get_stream(model_name, data_type, conditional=True):
    model = get_model(model_name)
    if data_type == 'train' or data_type == 'test':
        dataset = get_dataset(get_model_data(model_name), split=True)
        x = dataset[data_type + '_x']
        y = dataset[data_type + '_y']
        ranges = dataset['ranges']
    # elif data_type == 'sample_train' or 'sample_test':
    #     pass
    else:
        raise ValueError('Unknown data type {}. Should be one of [train, test, sample_train, sample_test]'.format(data_type))
    streams = compute_streams(model, x, y, ranges, conditional)
    return jsonify(streams)


def compute_stream(col_by_label, _range, bins=20):
    """Return a stream of shape [n_classes, n_bins]"""
    stream = []
    for col in col_by_label:
        hist, _ = np.histogram(col, bins, _range)
        stream.append(hist)
    return stream


def _compute_streams(x: np.ndarray, idx_by_label: List[np.ndarray], ranges: np.ndarray, bins=20) -> List[np.ndarray]:
    streams = []
    for col, _range in zip(x.T, ranges):
        col_by_label = [col[idx] for idx in idx_by_label]
        streams.append(compute_stream(col_by_label, _range, bins))
    return streams


def compute_streams(model: Union[RuleList, Tree], x, y, ranges, conditional=True, bins=20):
    """
    :param model: a RuleList or a Tree
    :param x: the data, of shape [n_instances, n_features]
    :param y: the target np.ndarray of shape [n_instances,]
    :param ranges: the ranges of each feature, of shape [n_features, 2], used for uniform binning
    :param conditional: whether to compute the stream conditional on previous rules/nodes, default to True
    :param bins: number of bins to compute the stream, default to 20
    :return:
        if not conditional, return a list of n_features streams,
        each stream is a np.ndarray of shape [n_classes, n_bins].
        if conditional, return a 2D np.ndarray of streams, the array is of shape [n_rules/n_nodes, n_features]
    """
    idx_by_label = []
    for label in range(model.n_classes):
        idx_by_label.append(y == label)
    if not conditional:
        return _compute_streams(x, idx_by_label, ranges, bins)

    if isinstance(model, RuleList):
        decision_supports = model.decision_support(x, transform=True)
        streams = []
        # supports per rule: a bool array of shape [n_instances,]
        for support in decision_supports:
            local_idx_by_label = [np.logical_and(support, idx) for idx in idx_by_label]
            streams.append(_compute_streams(x, local_idx_by_label, ranges, bins))
        return streams

    if isinstance(model, Tree):
        decision_supports = model.decision_support(x, transform=True)
        streams = []
        for i in range(model.n_nodes):
            row = decision_supports.getrow(i)
            local_idx_by_label = [np.logical_and(row, idx) for idx in idx_by_label]
            streams.append(_compute_streams(x, local_idx_by_label, ranges, bins))
        # for support in decision_supports:
        #     local_idx_by_label = [np.logical_and(support, idx) for idx in idx_by_label]
        #     streams.append(_compute_streams(x, local_idx_by_label, ranges, bins))
        return streams
    raise ValueError("Unknown model type {}".format(model.__class__))

