from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris

# from vendors.mdlpc import MDLPDiscretizer
from mdlp.discretization import MDLP
from iml.models import Tree, NeuralNet, load_model, RuleSurrogate, TreeSurrogate
from iml.data_processing import split_data, get_dataset
from iml.utils.io_utils import get_path


def prep_data(dataset='breast_cancer', discretize=False):
    if dataset == 'breast_cancer':
        data = load_breast_cancer()
    elif dataset == 'iris':
        data = load_iris()
    else:
        raise ValueError("Unknown dataset {}!".format(dataset))
    x = data['data']
    y = data['target']
    if discretize:
        x = MDLP().fit_transform(x, y)
    train_x, test_x, train_y, test_y = split_data(x, y, train_size=0.8, shuffle=False)
    # train_x, train_y = zip(*train_data)
    # test_x, test_y = zip(*test_data)
    feature_names = data['feature_names']  # type: List[str]
    feature_names = [feature_name.replace(' ', '_') for feature_name in feature_names]
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), feature_names


def train_tree(name='tree', dataset='wine'):
    data = get_dataset(dataset, split=True, discrete=False)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']
    model_name = '-'.join([dataset, name])
    tree = Tree(name=model_name, max_depth=6, min_samples_leaf=2)
    tree.train(train_x, train_y)
    tree.test(test_x, test_y)
    tree.describe()
    tree.export(get_path('models', '{}.json'.format(model_name)))
    tree.save()


def train_nn(name='nn', dataset='wine', neurons=(20,), **kwargs):
    data = get_dataset(dataset, split=True, discrete=False)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']
    model_name = '-'.join([dataset, name] + [str(neuron) for neuron in neurons])
    nn = NeuralNet(name=model_name, neurons=neurons, activation="relu",
                   alpha=0.01, max_iter=5000, verbose=True, **kwargs)
    nn.train(train_x, train_y)
    nn.test(test_x, test_y)
    nn.save()


def train_rule(name='rule', dataset='breast_cancer'):
    data = get_dataset(dataset, split=True, discrete=True)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']
    from iml.models.rule_model import SBRL

    # print(train_x.shape, train_x.dtype)
    discretizer = data['discretizer']
    model_name = '-'.join([dataset, name])
    brl = SBRL(name=model_name, rule_maxlen=3, discretizer=discretizer)
    brl.train(train_x, train_y)
    # print(brl.infer(test_x))
    brl.test(test_x, test_y)
    brl.describe(feature_names=feature_names)
    brl.save()


def train_surrogate(model_file, is_global=True, sampling_rate=5, surrogate='rule'):
    model = load_model(model_file)
    dataset = model.name.split('-')[0]
    data = get_dataset(dataset, split=True, discrete=True)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']
    # print(feature_names)
    model.test(test_x, test_y)

    model_name = surrogate + '-surrogate-' + model.name
    if surrogate == 'rule':
        surrogate_model = RuleSurrogate(name=model_name, discretizer=data['discretizer'],
                                        rule_minlen=1, rule_maxlen=3, min_support=0.02,
                                        _lambda=100, nchain=40, eta=1, iters=5000)
    elif surrogate == 'tree':
        surrogate_model = TreeSurrogate(name=model_name, max_depth=10, min_samples_leaf=3)
    else:
        raise ValueError("Unknown surrogate type {}".format(surrogate))
    if is_global:
        sigmas = np.std(train_x, axis=0) / (len(train_x) ** 0.7)
    else:
        sigmas = np.std(train_x, axis=0) / np.sqrt(len(train_x))
    # sigmas = [0] * train_x.shape[1]
    # print(sigmas)
    if is_global:
        instances = train_x
    else:
        instances = train_x[19:20, :]
    # print('train_y:')
    # print(train_y)
    # print('target_y')
    # print(model.predict(instances))
    surrogate_model.surrogate(model, instances, sigmas, sampling_rate * len(instances), discretize=True)
    surrogate_model.describe(feature_names=feature_names)
    surrogate_model.save()
    surrogate_model.self_test()
    if is_global:
        surrogate_model.test(test_x, test_y)
    else:
        surrogate_model.test(train_x[19:20, :], train_y[19:20])


if __name__ == '__main__':
    # train_tree(dataset='breast_cancer')
    # train_tree(dataset='wine')
    # train_tree(dataset='iris')
    # train_rule(dataset='breast_cancer')
    # train_rule(dataset='iris')
    # train_rule(dataset='wine')
    # train_nn(dataset='breast_cancer')
    # train_nn(dataset='iris')
    # train_nn(dataset='wine')
    # train_surrogate('models/iris-nn-20.mdl', surrogate='tree')
    # train_surrogate('models/breast_cancer-nn-20.mdl', surrogate='tree')
    # train_surrogate('models/wine-nn-20.mdl', surrogate='tree')
    # train_nn(dataset='diabetes', neurons=(50, 50), standardize=False,
    #          solver='sgd', momentum=0.8)

    train_nn(dataset='diabetes', neurons=(200, 100, 50), standardize=False,
             tol=1e-6)
