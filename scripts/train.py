from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris

# from vendors.mdlpc import MDLPDiscretizer
from mdlp.discretization import MDLP
from iml.models import Tree, NeuralNet, load_model
from iml.data_processing import split


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
        x = MDLP().fit_transform(x,y)
    train_data, test_data = split(list(zip(x, y)), [0.8, 0.2], shuffle=False)
    train_x, train_y = zip(*train_data)
    test_x, test_y = zip(*test_data)
    feature_names = data['feature_names']  # type: List[str]
    feature_names = [feature_name.replace(' ', '_') for feature_name in feature_names]
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), feature_names


def train_tree(name='tree'):
    train_x, train_y, test_x, test_y, feature_names = prep_data()
    tree = Tree(name=name, max_depth=5, min_samples_leaf=3)
    tree.train(train_x, train_y)
    tree.test(test_x, test_y)
    tree.describe()
    tree.export('{}.json'.format(name))
    tree.save()


def train_nn(neurons=(20,)):
    train_x, train_y, test_x, test_y, feature_names = prep_data()
    name = '-'.join(['nn']+[str(neuron) for neuron in neurons])
    nn = NeuralNet(name=name, neurons=neurons, activation="relu", alpha=0.01, max_iter=2000, solver='adam')
    nn.train(train_x, train_y)
    nn.test(test_x, test_y)
    nn.save()


def train_rule(name='rule'):
    train_x, train_y, test_x, test_y, feature_names = prep_data()
    from iml.models.rule_model import SBRL

    # print(train_x.shape, train_x.dtype)
    brl = SBRL(rule_maxlen=1)
    brl.train(train_x, train_y)
    # print(brl.infer(test_x))
    brl.test(test_x, test_y)
    brl.describe(feature_names=feature_names)
    brl.save()


def rule_surrogate(model_file, name='rule-s-nn', is_global=True, sampling_rate=10):
    from iml.models.rule_model import RuleSurrogate
    model = load_model(model_file)
    train_x, train_y, test_x, test_y, feature_names = prep_data()
    # print(feature_names)
    model.test(test_x, test_y)
    rule_model = RuleSurrogate(name=name, rule_maxlen=2,
                               minsupport_pos=0.01, minsupport_neg=0.01,
                               _lambda=400, nchain=40, eta=1)
    rule_model.discretizer.fit(train_x, train_y)
    if is_global:
        sigmas = np.std(train_x, axis=0)/ (len(train_x) ** 0.7)
    else:
        sigmas = np.std(train_x, axis=0)/np.sqrt(len(train_x))
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
    rule_model.surrogate(model, instances, sigmas, sampling_rate*len(instances), rediscretize=False)
    rule_model.describe(feature_names=feature_names)
    rule_model.save()
    rule_model.self_test()
    if is_global:
        rule_model.test(test_x, test_y)
    else:
        rule_model.test(train_x[19:20, :], train_y[19:20])


if __name__ == '__main__':
    train_rule()