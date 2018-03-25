import time

import numpy as np

from iml.models import Tree, NeuralNet, SVM, load_model, RuleSurrogate, TreeSurrogate, create_constraints
from iml.data_processing import get_dataset, sample_balance
from iml.utils.io_utils import get_path, dict2json, file_exists, json2dict

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
    acc, loss, auc = nn.test(test_x, test_y)
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
    acc, loss, auc = svm.test(test_x, test_y)
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
                    rule_maxlen=2, min_support=0.01, eta=1, _lambda=50, iters=50000, alpha=1):
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
                                        _lambda=_lambda, nchain=30, eta=eta, iters=iters, alpha=alpha)
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
        surrogate_model.surrogate(model, instances, constraints, sampling_rate, cov_factor=0.5, rediscretize=True)
    else:
        surrogate_model.surrogate(model, instances, constraints, sampling_rate)
    # surrogate_model.evaluate(train_x, train_y)
    # surrogate_model.describe(feature_names=feature_names)
    surrogate_model.save()
    # surrogate_model.self_test()
    self_fidelity = surrogate_model.self_test(len(train_y) * sampling_rate * 0.25)
    fidelity, acc = surrogate_model.test(test_x, test_y)
    return fidelity, acc, self_fidelity, surrogate_model.n_rules


# datasets = ['breast_cancer', 'wine', 'iris', 'wine_quality_red', 'abalone', 'adult']
datasets = ['wine_quality_red']


def train_all_nn():
    layers = [1, 2, 3, 4]
    neurons = 40
    alphas = [0.01, 0.1, 1.0]

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
    cs = [0.1, 1, 10]

    # NN
    names = []
    for dataset in datasets:
        # model_names = []
        model_name = '-'.join([dataset, 'svm'])
        if file_exists(get_path('models', model_name + '.mdl')):
            names.append(model_name)
            continue
        best_svm = None
        score = 0
        for c in cs:
            model, acc = train_svm(dataset=dataset, C=c)
            if acc > score:
                score = acc
                best_svm = model
        best_svm.save()
        names.append(best_svm.name)
        # names.append(model_names)
    return names


def run_test(dataset, names, n_test=10, alpha=1):
    results = []
    sampling_rate = 2 if dataset in {'adult'} else 5.
    rule_maxlen = 3
    _lambda = 10 if dataset in {'iris', 'breast_cancer', 'wine'} else 70
    for name in names:
        model_file = get_path('models', name + '.mdl')
        fidelities = []
        self_fidelities = []
        accs = []
        seconds = []
        list_lengths = []
        for i in range(n_test):
            print('test', i)
            start = time.time()
            fidelity, acc, self_fidelity, n_rules = train_surrogate(model_file, surrogate='rule',
                                                                    sampling_rate=sampling_rate, iters=100000,
                                                                    rule_maxlen=rule_maxlen, alpha=alpha,
                                                                    min_support=0.02, _lambda=_lambda)
            seconds.append(time.time() - start)
            print('time: {}s; length: {}'.format(seconds[-1], n_rules))
            list_lengths.append(n_rules)
            self_fidelities.append(self_fidelity)
            fidelities.append(fidelity)
            accs.append(acc)

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
        mean_time = float(np.mean(seconds))
        mean_length = float(np.mean(list_lengths))
        obj = {'fidelity': fidelities, 'acc': accs,
               'std_acc': std_acc, 'mean_acc': mean_acc, 'min_acc': min_acc, 'max_acc': max_acc,
               'std_fidelity': std_fidelity, 'mean_fidelity': mean_fidelity,
               'min_fidelity': min_fidelity, 'max_fidelity': max_fidelity,
               'std_self_fidelity': std_self_fidelity, 'mean_self_fidelity': mean_self_fidelity,
               'min_self_fidelity': min_self_fidelity, 'max_self_fidelity': max_self_fidelity,
               'time': seconds, 'mean_time': mean_time, 'lengths': list_lengths, 'mean_length': mean_length}
        print(dataset)
        print(name)
        print(obj)
        print('---------')
        results.append(obj)
    return results


def test(target='svm'):

    n_test = 10
    # max_rulelens = [2, 2, 2, 3, 3, 3]
    performance_dict = {}
    if target == 'nn':
        nns = train_all_nn()
        for i, nn_names in enumerate(nns):
            dataset = datasets[i]
            # max_rulelen = max_rulelens[i]
            # performance_dict
            file_name = get_path('experiments', dataset + '-nn.json')
            if file_exists(file_name):
                results = json2dict(file_name)
            else:
                results = run_test(dataset, nn_names, n_test=n_test, alpha=0)
                dict2json(results, file_name)
            performance_dict[dataset] = results
 
        dict2json(performance_dict, 'results-nn.json')
        return
    performance_dict = {}
    svms = train_all_svm()
    print(svms)
    for i, svm_name in enumerate(svms):
        dataset = datasets[i]
        # max_rulelen = max_rulelens[i]
        # performance_dict
        file_name = get_path('experiments', dataset + '-svm.json')
        if file_exists(file_name):
            results = json2dict(file_name)
        else:
            results = run_test(dataset, [svm_name], n_test=n_test, alpha=0)
            dict2json(results, file_name)
        performance_dict[dataset] = results

    dict2json(performance_dict, 'results-svm.json')


def test_sampling_rate(dataset='abalone3'):
    n_test = 10
    # max_rulelens = [2, 2, 2, 3, 3, 3]
    neurons = (50, 50, 50, 50)
    sampling_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    performance_dict = {}
    model_name = '-'.join([dataset, 'nn'] + [str(i) for i in neurons])
    for sampling_rate in sampling_rates:
        file_name = get_path('experiments', '-'.join(['sample', dataset, str(sampling_rate), '.json']))
        if file_exists(file_name):
            results = json2dict(file_name)
        else:
            results = run_test(dataset, [model_name], n_test=n_test)
            dict2json(results, file_name)
        performance_dict[dataset] = results

    dict2json(performance_dict, '-'.join(['sample', dataset, '.json']))
    return


if __name__ == '__main__':

    test('nn')
    # test_sampling_rate('abalone3')
    #
    # nn, acc = train_nn(dataset='abalone3', neurons=(50, 50, 50, 50), alpha=0.01, tol=1e-6)
    # nn.save()
