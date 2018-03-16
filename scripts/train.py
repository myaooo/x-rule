from typing import List

import numpy as np

from iml.models import Tree, NeuralNet, SVM, load_model, RuleSurrogate, TreeSurrogate, create_constraints
from iml.data_processing import get_dataset, sample_balance
from iml.utils.io_utils import get_path

rebalance = False


def get_constraints(data, is_categorical):
    sigmas = np.std(data, axis=0) / (len(data) ** 0.5)
    ranges = [None] * len(sigmas)
    return create_constraints(is_categorical, np.logical_not(is_categorical), ranges)


def train_tree(name='tree', dataset='wine', max_depth=None, min_samples_leaf=0.005, **kwargs):
    data = get_dataset(dataset, split=True, discrete=False, one_hot=True)
    train_x, train_y, test_x, test_y, feature_names, one_hot_encoder = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names'], data['one_hot_encoder']

    if rebalance:
        print("balancing training data")
        train_x, train_y = sample_balance(train_x, train_y)
        print("#data after balancing:", len(train_y))

    model_name = '-'.join([dataset, name])
    tree = Tree(name=model_name, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                one_hot_encoder=one_hot_encoder, **kwargs)
    tree.train(train_x, train_y)
    tree.evaluate(train_x, train_y, stage='train')
    tree.test(test_x, test_y)
    tree.describe()
    tree.export(get_path('models', '{}.json'.format(model_name)))
    tree.save()


def train_nn(name='nn', dataset='wine', neurons=(20,), alpha=0.01, problem='classification', **kwargs):
    data = get_dataset(dataset, split=True, discrete=False, one_hot=True)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']
    if rebalance:
        print("balancing training data")
        train_x, train_y = sample_balance(train_x, train_y)
        print("#data after balancing:", len(train_y))
    one_hot_encoder, is_categorical = data['one_hot_encoder'], data['is_categorical']
    model_name = '-'.join([dataset, name] + [str(neuron) for neuron in neurons])
    nn = NeuralNet(name=model_name, problem=problem, neurons=neurons, max_iter=5000, alpha=alpha,
                   one_hot_encoder=one_hot_encoder, **kwargs)
    nn.train(train_x, train_y)
    nn.evaluate(train_x, train_y, stage='train')
    nn.test(test_x, test_y)
    nn.save()


def train_svm(name='svm', dataset='wine', C=1.0, problem='classification', **kwargs):
    data = get_dataset(dataset, split=True, discrete=False, one_hot=True)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']
    if rebalance:
        print("balancing training data")
        train_x, train_y = sample_balance(train_x, train_y)
        print("#data after balancing:", len(train_y))
    one_hot_encoder, is_categorical = data['one_hot_encoder'], data['is_categorical']
    model_name = '-'.join([dataset, name])
    svm = SVM(name=model_name, problem=problem, C=C, one_hot_encoder=one_hot_encoder, **kwargs)
    svm.train(train_x, train_y)
    svm.evaluate(train_x, train_y, stage='train')
    svm.test(test_x, test_y)
    svm.save()


def train_rule(name='rule', dataset='breast_cancer', rule_max_len=2, **kwargs):
    data = get_dataset(dataset, split=True, discrete=True)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']
    from iml.models.rule_model import RuleList

    if rebalance:
        print("balancing training data")
        train_x, train_y = sample_balance(train_x, train_y)
        print("#data after balancing:", len(train_y))

    # print(train_x.shape, train_x.dtype)
    discretizer = data['discretizer']
    model_name = '-'.join([dataset, name])
    brl = RuleList(name=model_name, rule_maxlen=rule_max_len, discretizer=discretizer, **kwargs)
    brl.train(train_x, train_y)
    brl.evaluate(train_x, train_y, stage='train')
    # print(brl.infer(test_x))
    brl.test(test_x, test_y)
    brl.describe(feature_names=feature_names)
    brl.save()


def train_surrogate(model_file, is_global=True, sampling_rate=5, surrogate='rule',
                    rule_maxlen=2, min_support=0.01, eta=1):
    is_rule = surrogate == 'rule'
    model = load_model(model_file)
    dataset = model.name.split('-')[0]
    data = get_dataset(dataset, split=True, discrete=is_rule, one_hot=is_rule)
    train_x, train_y, test_x, test_y, feature_names, is_categorical = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names'], data['is_categorical']
    # print(feature_names)
    print("Original model:")
    model.test(test_x, test_y)
    print("Surrogate model:")

    model_name = surrogate + '-surrogate-' + model.name
    if surrogate == 'rule':
        surrogate_model = RuleSurrogate(name=model_name, discretizer=data['discretizer'],
                                        rule_minlen=1, rule_maxlen=rule_maxlen, min_support=min_support,
                                        _lambda=50, nchain=30, eta=eta, iters=50000)
    elif surrogate == 'tree':
        surrogate_model = TreeSurrogate(name=model_name, max_depth=None, min_samples_leaf=0.01)
    else:
        raise ValueError("Unknown surrogate type {}".format(surrogate))
    constraints = get_constraints(train_x, is_categorical)
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
    if isinstance(surrogate_model, RuleSurrogate):
        surrogate_model.surrogate(model, instances, constraints, sampling_rate, rediscretize=True)
    else:
        surrogate_model.surrogate(model, instances, constraints, sampling_rate)
    # surrogate_model.evaluate(train_x, train_y)
    surrogate_model.describe(feature_names=feature_names)
    surrogate_model.save()
    # surrogate_model.self_test()
    if is_global:
        surrogate_model.test(test_x, test_y)
    else:
        surrogate_model.test(train_x[19:20, :], train_y[19:20])


if __name__ == '__main__':

    ###########
    # Trees
    ###########

    # train_tree(dataset='breast_cancer')
    # train_tree(dataset='wine')
    # train_tree(dataset='iris')
    # train_tree(dataset='thoracic')
    # train_tree(dataset='bank_marketing')
    # train_tree(dataset='credit_card')
    # train_tree(dataset='adult')
    # train_tree(dataset='abalone', min_samples_leaf=0.01)
    # train_tree(dataset='wine_quality_red', min_samples_leaf=0.01)
    # train_tree(dataset='wine_quality_white', min_samples_leaf=0.01)
    # train_tree(dataset='diabetes', min_samples_leaf=25, max_depth=None)

    ###########
    # Rules
    ###########

    # train_rule(dataset='breast_cancer')
    # train_rule(dataset='iris')
    # train_rule(dataset='wine')
    # train_rule(dataset='thoracic', rule_max_len=2, )
    # train_rule(dataset='bank_marketing')
    # train_rule(dataset='abalone', min_support=0.01, eta=2)
    # train_rule(dataset='credit_card')
    # train_rule(dataset='adult')
    # train_rule(dataset='wine_quality_red', min_support=0.01, eta=2)
    # train_rule(dataset='wine_quality_white', min_support=0.01, eta=2)

    ###########
    # SVMs
    ###########

    # train_svm(dataset='breast_cancer')
    # train_svm(dataset='iris')
    # train_svm(dataset='wine')
    # train_svm(dataset='thoracic')
    # train_svm(dataset='bank_marketing')
    # train_svm(dataset='abalone')
    # train_svm(dataset='credit_card')
    # train_svm(dataset='adult')
    # train_svm(dataset='wine_quality_red')
    # train_svm(dataset='wine_quality_white')

    ###########
    # NNs
    ###########

    # train_nn(dataset='breast_cancer')
    # train_nn(dataset='iris')
    # train_nn(dataset='wine')
    # train_nn(dataset='thoracic', neurons=(30, 30), alpha=5.0)
    # train_nn(dataset='bank_marketing', neurons=(30, 30), alpha=1.0)
    # train_nn(dataset='credit_card', neurons=(40, 40), alpha=0.01)
    # train_nn(dataset='abalone', neurons=(40, 40), tol=1e-6, alpha=0.01, verbose=True)
    # train_nn(dataset='adult', neurons=(30, 31), alpha=0.1)
    # train_nn(dataset='diabetes', neurons=(50, 50), standardize=False,
    #          solver='sgd', momentum=0.8)
    # train_nn(dataset='diabetes', neurons=(50,), verbose=True, tol=1e-6)
    # train_nn(dataset='diabetes', neurons=(200, 100, 50), verbose=True,
    #          tol=1e-6)
    # train_nn(dataset='wine_quality_red', neurons=(40, 40), tol=1e-6, alpha=0.01, activation='logistic', verbose=True)
    # train_nn(dataset='wine_quality_white', neurons=(40, 40), tol=1e-6, alpha=0.01, activation='logistic', verbose=True)

    ###########
    # Surrogates of NNs
    ###########

    # train_surrogate('models/abalone-nn-40-40-40.mdl', surrogate='rule')
    # train_surrogate('models/abalone-nn-30-30-30-30.mdl', surrogate='tree')

    # train_surrogate('models/iris-nn-20.mdl', surrogate='rule')
    # train_surrogate('models/breast_cancer-nn-20.mdl', surrogate='rule')
    # train_surrogate('models/wine-nn-20.mdl', surrogate='rule')

    # train_surrogate('models/diabetes-nn-200-100-50.mdl', surrogate='tree', sampling_rate=0.1)
    # train_surrogate('models/adult-nn-30-30.mdl', surrogate='rule', sampling_rate=5)
    train_surrogate('models/wine_quality_red-nn-40-40.mdl', surrogate='rule', sampling_rate=5, rule_maxlen=3)
    # train_surrogate('models/wine_quality_white-nn-40-40.mdl', surrogate='rule', sampling_rate=10)

    ###########
    # Surrogates of SVMs
    ###########

    # train_surrogate('models/iris-svm.mdl', surrogate='rule')
    # train_surrogate('models/breast_cancer-svm.mdl', surrogate='rule')
    # train_surrogate('models/wine-svm.mdl', surrogate='rule')
    # train_surrogate('models/abalone-svm.mdl', surrogate='rule', sampling_rate=2)
    # train_surrogate('models/bank_marketing-svm.mdl', surrogate='rule', sampling_rate=2)
    # train_surrogate('models/thoracic-svm.mdl', surrogate='rule', sampling_rate=2)
    # train_surrogate('models/adult-svm.mdl', surrogate='rule', sampling_rate=2)
    # train_surrogate('models/wine_quality_red-svm.mdl', surrogate='rule', sampling_rate=5)
    # train_surrogate('models/wine_quality_white-svm.mdl', surrogate='rule', sampling_rate=5)

    # train_surrogate('models/diabetes-nn-200-100-50.mdl', surrogate='tree', sampling_rate=0.1)
