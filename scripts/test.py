from typing import List

import numpy as np

from iml.models import Tree, NeuralNet, SVM, load_model, RuleSurrogate, TreeSurrogate, create_constraints
from iml.data_processing import get_dataset, sample_balance
from iml.utils.io_utils import get_path, dict2json, file_exists

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
    loss, acc, auc = nn.test(test_x, test_y)
    return nn, acc


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
    loss, acc, auc = svm.test(test_x, test_y)
    return svm, acc


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


def train_surrogate(model_file, sampling_rate=5, surrogate='rule',
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
    instances = train_x
    # print('train_y:')
    # print(train_y)
    # print('target_y')
    # print(model.predict(instances))
    if isinstance(surrogate_model, RuleSurrogate):
        surrogate_model.surrogate(model, instances, constraints, sampling_rate, rediscretize=True)
    else:
        surrogate_model.surrogate(model, instances, constraints, sampling_rate)
    # surrogate_model.evaluate(train_x, train_y)
    # surrogate_model.describe(feature_names=feature_names)
    # surrogate_model.save()
    # surrogate_model.self_test()
    self_fidelity = surrogate_model.self_test(len(train_y) * 0.25)
    fidelity, acc = surrogate_model.test(test_x, test_y)
    return fidelity, acc, self_fidelity


datasets = ['breast_cancer', 'wine', 'iris', 'adult', 'wine_quality_red']


def train_all_nn():
    layers = [1, 2, 3, 4]
    neurons = 40
    alphas = [0.001, 0.01, 0.1]

    # NN
    names = []
    for dataset in datasets:
        model_names = []
        for layer in layers:
            hidden_layers = [neurons] * layer
            model_name = '-'.join([dataset, 'nn'] + [str(neuron) for neuron in hidden_layers])
            if file_exists(get_path('models', model_name + '.mdl')):
                model_names.append(model_name)
                continue
            best_nn = None
            score = 0
            for alpha in alphas:
                nn, acc = train_nn(dataset=dataset, neurons=hidden_layers, alpha=alpha, tol=1e-6)
                if acc > score:
                    score = acc
                    best_nn = nn
            best_nn.save()
            model_names.append(best_nn.name)
        names.append(model_names)
    return names


def train_all_svm():
    cs = [0.01, 0.1, 1, 10]

    # NN
    names = []
    for dataset in datasets:
        model_names = []
        model_name = '-'.join([dataset, 'svm'])
        if file_exists(get_path('models', model_name + '.mdl')):
            model_names.append(model_name)
            continue
        best_svm = None
        score = 0
        for c in cs:
            model, acc = train_svm(dataset=dataset, C=c)
            if acc > score:
                score = acc
                best_svm = model
        best_svm.save()
        model_names.append(best_svm.name)
        names.append(model_names)
    return names


def run_test(dataset, names, rule_maxlen, n_test=10):
    results = []
    for name in names:
        model_file = get_path('models', name + '.mdl')
        fidelities = []
        self_fidelities = []
        accs = []
        for i in range(n_test):
            print('test', i)
            try:
                fidelity, acc, self_fidelity = train_surrogate(model_file, surrogate='rule', sampling_rate=5,
                                                               rule_maxlen=rule_maxlen)
                self_fidelities.append(self_fidelity)
                fidelities.append(fidelity)
                accs.append(acc)
            except:
                print('error occurs')
                print('just keep move on')

        std_acc = float(np.std(accs))
        mean_acc = float(np.mean(accs))
        max_acc = float(np.max(accs))
        min_acc = float(np.min(accs))
        std_fidelity = float(np.std(fidelities))
        mean_fidelity = float(np.mean(fidelities))
        max_fidelity = float(np.max(fidelities))
        min_fidelity = float(np.min(fidelities))
        std_self_fidelity = float(np.std(self_fidelities))
        mean_self_fidelity = float(np.mean(self_fidelities))
        max_self_fidelity = float(np.max(self_fidelities))
        min_self_fidelity = float(np.min(self_fidelities))
        obj = {'fidelity': fidelities, 'acc': accs,
               'std_acc': std_acc, 'mean_acc': mean_acc, 'min_acc': min_acc, 'max_acc': max_acc,
               'std_fidelity': std_fidelity, 'mean_fidelity': mean_fidelity,
               'min_fidelity': min_fidelity, 'max_fidelity': max_fidelity,
               'std_self_fidelity': std_self_fidelity, 'mean_self_fidelity': mean_self_fidelity,
               'min_self_fidelity': min_self_fidelity, 'max_self_fidelity': max_self_fidelity}
        print(dataset)
        print(name)
        print(obj)
        print('---------')
        results.append(obj)
    return results


def test():

    n_test = 10
    nns = train_all_nn()
    svms = train_all_svm()
    max_rulelens = [2, 2, 2, 3, 3]
    performance_dict = {}
    for i, nn_names in enumerate(nns):
        dataset = datasets[i]
        max_rulelen = max_rulelens[i]
        # performance_dict
        results = run_test(dataset, nn_names, max_rulelen, n_test=n_test)
        dict2json(results, dataset + '-nn.json')
        performance_dict[dataset] = results

    dict2json(performance_dict, 'results-nn.json')

    performance_dict = {}
    for i, svm_names in enumerate(svms):
        dataset = datasets[i]
        max_rulelen = max_rulelens[i]
        # performance_dict
        results = run_test(dataset, svm_names, max_rulelen, n_test=n_test)
        dict2json(results, dataset + '-svm.json')
        performance_dict[dataset] = results

    dict2json(performance_dict, 'results-svm.json')


if __name__ == '__main__':

    test()
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
    # train_surrogate('models/wine_quality_red-nn-40-40.mdl', surrogate='rule', sampling_rate=5, rule_maxlen=3)
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
