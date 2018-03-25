
from functools import reduce
import numpy as np

from iml.data_processing import get_dataset
from iml.utils.io_utils import dict2json
from iml.models import SKClassifier, RuleSurrogate, create_constraints, load_model


def get_constraints(is_categorical, ranges):
    # sigmas = np.std(data, axis=0) / (len(data) ** 0.5)
    # ranges = [None] * len(sigmas)
    return create_constraints(is_categorical, np.logical_not(is_categorical), ranges)


def over_sampling(x, y, filters, rate=2.):
    if rate < 1:
        raise ValueError("rate must larger than 1")
    satisfied = np.ones(len(y), dtype=np.bool)
    for idx, interval in filters.items():
        assert isinstance(interval, list)
        if idx == x.shape[1]:
            col = y
            logics = [col == i for i in interval]
            filtered = reduce(np.logical_or, logics)
        else:
            col = x[:, idx]
            filtered = np.logical_and(interval[0] < col, col < interval[1])
        satisfied = np.logical_and(satisfied, filtered)
    x_filtered = x[satisfied, :]
    y_filtered = y[satisfied]
    n_sampling = int(len(y_filtered) * (rate - 1))
    sample_indices = np.random.randint(len(y_filtered), size=n_sampling)
    x = np.vstack((x, x_filtered[sample_indices]))
    y = np.concatenate((y, y_filtered[sample_indices]))

    shuffle_idx = np.arange(len(y))
    np.random.shuffle(shuffle_idx)

    return x[shuffle_idx], y[shuffle_idx]


def re_sampling(x, y, filters, rate=2.):
    # if rate < 1:
    #     raise ValueError("rate must larger than 1")
    satisfied = np.ones(len(y), dtype=np.bool)
    for idx, interval in filters.items():
        assert isinstance(interval, list)
        if idx == x.shape[1]:
            col = y
            logics = [col == i for i in interval]
            filtered = reduce(np.logical_or, logics)
        else:
            col = x[:, idx]
            filtered = np.logical_and(interval[0] < col, col < interval[1])
        satisfied = np.logical_and(satisfied, filtered)
    x_kept = x[np.logical_not(satisfied), :]
    y_kept = y[np.logical_not(satisfied)]
    x_filtered = x[satisfied, :]
    y_filtered = y[satisfied]
    times = int(rate)
    n_sampling = int(len(y_filtered) * (rate - times))
    sample_indices = np.tile(np.arange(len(y_filtered)), times)
    sample_indices = np.concatenate((sample_indices, np.random.randint(len(y_filtered), size=n_sampling)))
    x = np.vstack((x_kept, x_filtered[sample_indices]))
    y = np.concatenate((y_kept, y_filtered[sample_indices]))

    shuffle_idx = np.arange(len(y))
    np.random.shuffle(shuffle_idx)

    return x[shuffle_idx], y[shuffle_idx]


filters1 = {'Glucose': [105, 121], 'Age': [31.5, 64.4], 'Body Mass Index': [25.7, 100]}
filters2 = {'Glucose': [108, 138], 'Age': [31.9, 100], 'Body Mass Index': [25.9, 100],
            'Diabetes Pedigree Function': [0., 1.18]}


def do_re_sample(train_x, train_y, feature_names):
    uniq, counts = np.unique(train_y, return_counts=True)
    print('before sample: [{}]'.format('/'.join([str(c) for c in counts])))

    sample_filters = filters2

    filters = {feature_names.index(key): value for key, value in sample_filters.items()}
    # filters[train_x.shape[1]] = [0]
    # filters
    print("over sampling training data")
    train_x, train_y = over_sampling(train_x, train_y, filters, rate=1.5)
    print("#data after over sampling:", len(train_y))

    uniq, counts = np.unique(train_y, return_counts=True)
    print('after sample: [{}]'.format('/'.join([str(c) for c in counts])))
    return train_x, train_y


def train_nn(name='sample', dataset='pima', neurons=(20,), alpha=0.01, sample=True, **kwargs):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    data = get_dataset(dataset, split=True, discrete=False, one_hot=True)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']

    # filters[train_x.shape[1]] = [0]
    # filters
    if sample:
        train_x, train_y = do_re_sample(train_x, train_y, feature_names)

    one_hot_encoder, is_categorical = data['one_hot_encoder'], data['is_categorical']
    model_name = '-'.join([dataset, 'nn'] + [str(neuron) for neuron in neurons] + [name])
    model = MLPClassifier(hidden_layer_sizes=neurons, max_iter=5000, alpha=alpha, **kwargs)
    nn = SKClassifier(model, name=model_name, standardize=True, one_hot_encoder=one_hot_encoder)
    nn.train(train_x, train_y)
    nn.evaluate(train_x, train_y, stage='train')
    acc, loss, auc = nn.test(test_x, test_y)
    nn.save()
    return acc, loss, auc


def train_svm(name='oversample', dataset='pima', C=1., sample=True, **kwargs):
    from sklearn.svm import SVC
    data = get_dataset(dataset, split=True, discrete=False, one_hot=True)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']

    uniq, counts = np.unique(train_y, return_counts=True)
    print('before sample: [{}]'.format('/'.join([str(c) for c in counts])))

    sample_filters = {'Glucose': [105, 121], 'Age': [31.5, 64.4], 'Body Mass Index': [25.7, 100]}

    filters = {feature_names.index(key): value for key, value in sample_filters.items()}
    # filters[train_x.shape[1]] = [0]
    # filters
    print("over sampling training data")
    if sample:
        train_x, train_y = re_sampling(train_x, train_y, filters, rate=1)
    print("#data after over sampling:", len(train_y))

    uniq, counts = np.unique(train_y, return_counts=True)
    print('after sample: [{}]'.format('/'.join([str(c) for c in counts])))

    one_hot_encoder, is_categorical = data['one_hot_encoder'], data['is_categorical']
    model_name = '-'.join([dataset, 'svm'] + [name])
    model = SVC(C=C, probability=True, **kwargs)
    nn = SKClassifier(model, name=model_name, standardize=True, one_hot_encoder=one_hot_encoder)
    nn.train(train_x, train_y)
    nn.evaluate(train_x, train_y, stage='train')
    acc, loss, auc = nn.test(test_x, test_y)
    nn.save()
    return acc, loss, auc


def train_surrogate(model_file, sampling_rate=5., sample=True,
                    rule_maxlen=2, min_support=0.01, eta=1, iters=50000, _lambda=30):
    model = load_model(model_file)
    dataset = model.name.split('-')[0]
    data = get_dataset(dataset, split=True, discrete=True, one_hot=False)
    train_x, train_y, test_x, test_y, feature_names, is_categorical = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names'], data['is_categorical']
    ranges = data['ranges']

    if sample:
        train_x, train_y = do_re_sample(train_x, train_y, feature_names)
    # print(feature_names)
    print("Original model:")
    model.test(test_x, test_y)
    print("Surrogate model:")

    model_name = 'rule-surrogate-' + model.name
    surrogate_model = RuleSurrogate(name=model_name, discretizer=data['discretizer'],
                                    rule_minlen=1, rule_maxlen=rule_maxlen, min_support=min_support,
                                    _lambda=_lambda, nchain=30, eta=eta, iters=iters)
    constraints = get_constraints(is_categorical, ranges)
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
    surrogate_model.describe(feature_names=feature_names)
    surrogate_model.save()
    # surrogate_model.self_test()
    surrogate_model.test(test_x, test_y)


def main():
    # train_nn(dataset='abalone2', neurons=(40, 40), tol=1e-6, alpha=0.0001)
    accs = []
    losses = []
    aucs = []
    for i in range(10):
        # acc, loss, auc=train_nn(dataset='pima', neurons=(20, 20), tol=1e-5, alpha=1.0, sample=False, name='original')
        acc, loss, auc = train_nn(dataset='pima', neurons=(20, 20), tol=1e-5, alpha=1.0, sample=True, name='sample')
        # acc, loss, auc = train_svm(dataset='pima', C=0.1)
        accs.append(acc)
        losses.append(loss)
        aucs.append(auc)

    print(np.mean(accs))
    print(np.std(accs))
    print(np.min(accs))
    print(np.max(accs))
    dict2json({'loss': losses, 'acc': accs, 'auc': aucs}, 'case-pima-nn-2-sample2.json')


def cv_nn(dataset, neurons=(20,20), max_iter=1000):
    from sklearn.model_selection import cross_validate, ShuffleSplit
    from sklearn.neural_network import MLPClassifier

    n_test = 5
    data = get_dataset(dataset, split=True, discrete=False, one_hot=True)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']

    train_x, train_y = do_re_sample(train_x, train_y, feature_names)

    alphas = [0.1, 0.5, 1.0]
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    for alpha in alphas:
        clf = MLPClassifier(neurons, alpha=alpha, max_iter=max_iter, tol=1e-5)
        scores = []
        for i in range(n_test):
            cv_scores = cross_validate(clf, train_x, train_y, cv=cv)
            scores += cv_scores['test_score'].tolist()
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        print('alpha {}:'.format(alpha))
        print('score: {}, std: {}, min: {}, max: {}\n'.format(mean_score, std_score, min_score, max_score))


if __name__ == '__main__':
    # cv_nn('pima')
    main()
    # train_nn(dataset='pima', neurons=(20, 20), tol=1e-5, alpha=1.0, name='unsampled', sample=False)

    # train_surrogate('../models/pima-nn-10-10-unsampled.mdl', sample=False, min_support=0.05,
    #                 sampling_rate=5, rule_maxlen=3, eta=2, _lambda=10, iters=90000)

    # train_nn(dataset='pima', neurons=(10, 10), tol=1e-5, alpha=0.5, name='sample')

    # train_surrogate('../models/pima-nn-10-10-sample.mdl', min_support=0.05,
    #                 sampling_rate=10, rule_maxlen=3, eta=2, _lambda=20, iters=70000)
