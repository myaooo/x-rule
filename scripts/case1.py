
from functools import reduce
import numpy as np

from iml.data_processing import get_dataset
from iml.utils.io_utils import dict2json
from iml.models import SKClassifier


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
    sample_indice = np.random.randint(len(y_filtered), size=n_sampling)
    x = np.vstack((x, x_filtered[sample_indice]))
    y = np.concatenate((y, y_filtered[sample_indice]))

    shuffle_idx = np.arange(len(y))
    np.random.shuffle(shuffle_idx)

    return x[shuffle_idx], y[shuffle_idx]


def train_nn(name='nn', dataset='abalone2', neurons=(20,), alpha=0.01, **kwargs):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    data = get_dataset(dataset, split=True, discrete=False, one_hot=True)
    train_x, train_y, test_x, test_y, feature_names = \
        data['train_x'], data['train_y'], data['test_x'], data['test_y'], data['feature_names']

    uniq, counts = np.unique(train_y, return_counts=True)
    print('before sample: [{}]'.format('/'.join([str(c) for c in counts])))

    # abalone2
    sample_filters = {'shell weight': [0.249, 0.432], 'shucked weight': [0.337, 0.483]}

    # wine_quality_red
    # sample_filters = {'alcohol': [10.5, 11.7]}

    filters = {feature_names.index(key): value for key, value in sample_filters.items()}
    # abalone2
    # filters[train_x.shape[1]] = [0, 2]
    # wine_quality_red
    # filters[train_x.shape[1]] = [2, 4]

    # filters
    print("over sampling training data")
    train_x, train_y = over_sampling(train_x, train_y, filters, rate=2)
    print("#data after over sampling:", len(train_y))

    uniq, counts = np.unique(train_y, return_counts=True)
    print('after sample: [{}]'.format('/'.join([str(c) for c in counts])))

    one_hot_encoder, is_categorical = data['one_hot_encoder'], data['is_categorical']
    model_name = '-'.join([dataset, name] + [str(neuron) for neuron in neurons] + ['oversample'])
    model = MLPClassifier(hidden_layer_sizes=neurons, max_iter=5000, alpha=alpha, **kwargs)
    nn = SKClassifier(model, name=model_name, standardize=True, one_hot_encoder=one_hot_encoder)
    nn.train(train_x, train_y)
    nn.evaluate(train_x, train_y, stage='train')
    acc, loss, auc = nn.test(test_x, test_y)
    nn.save()
    return acc, loss, auc


def main():
    # train_nn(dataset='abalone2', neurons=(40, 40), tol=1e-6, alpha=0.0001)
    accs = []
    losses = []
    aucs = []
    for i in range(10):
        acc, loss, auc = train_nn(dataset='abalone2', neurons=(40,), tol=1e-6, alpha=0.001)
        accs.append(acc)
        losses.append(loss)
        aucs.append(auc)

    print(np.mean(accs))
    print(np.std(accs))
    print(np.mean(aucs))
    print(np.std(aucs))
    dict2json({'loss': losses, 'acc': accs, 'auc': aucs}, 'case1-abalone-no-sample.json')


if __name__ == '__main__':
    # main()
    train_nn(dataset='abalone2', neurons=(40,), tol=1e-6, alpha=0.001)
