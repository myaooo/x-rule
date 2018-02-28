from typing import List, Union, Tuple, Set, Iterable
import logging
import functools

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from mdlp.discretization import MDLP
from fim import fpgrowth, eclat

from iml.utils.io_utils import before_save, get_path, file_exists, load_file, save_file


_datasets_path = 'datasets/'
_cached_path = 'datasets/cached/'

# print(_datasets_path)

split_data = train_test_split


def add_cache_support(n_files=None):
    """
    A decorator that add cache functionality
    The decorator would add an argument `filenames` as the first input argument,
    which represents the cached filenames
    :param n_files: optional hinter on how many returned data/cached files
    :return:
    """
    # if isinstance(filenames, str):
    #     filenames = [filenames]
    def decorate(func):

        def ret_func_single(filename: str, *args, **kwargs):
            if not file_exists(filename):
                ret_data = func(*args, **kwargs)
                save_file(ret_data, filename)
                return ret_data
            else:
                return load_file(filename)

        def ret_func(filenames: List[str], *args, **kwargs):
            if n_files is not None:
                assert len(filenames) >= n_files
            all_file_exists = True
            for filename in filenames:
                if not file_exists(filename):
                    all_file_exists = False
                    break
            if all_file_exists:
                return (load_file(filename) for filename in filenames)
            else:
                ret_data = func(*args, **kwargs)
                assert len(filenames) >= len(ret_data)
                for data, filename in zip(ret_data, filenames):
                    before_save(filename)
                    save_file(data, filename)
                return ret_data

        def wrap_up(*args, **kwargs):
            if 'filenames' not in kwargs:
                return func(*args, **kwargs)
            filenames = kwargs['filenames']
            del kwargs['filenames']
            if isinstance(filenames, str):
                return ret_func_single(filenames, *args, **kwargs)
            return ret_func(filenames, *args, **kwargs)

        return wrap_up

    return decorate


_csv_files = ['target', 'data']
_json_files = ['feature_names', 'target_names', 'categories', 'is_categorical', 'is_binary']


def save_data(data, name):

    dataset_path = _datasets_path + name
    print("data saved to {}".format(dataset_path))
    for field in _csv_files:
        filename = field + '.csv'
        file_path = get_path(dataset_path, filename)
        before_save(file_path)
        save_file(data[field], file_path)

    descriptor = {key: data[key] for key in _json_files if key in data}
    save_file(descriptor, get_path(dataset_path, 'spec.json'))


def load_data(name):
    dataset_path = _datasets_path + name
    dataset = {}
    for field in _csv_files:
        filename = field + '.csv'
        file_path = get_path(dataset_path, filename)
        dataset[field] = load_file(file_path)

    for field in ['target']:
        dataset[field] = dataset[field].reshape((-1))

    descriptor = load_file(get_path(dataset_path, 'spec.json'))
    for key, val in descriptor.items():
        dataset[key] = val
    return dataset


sklearn_datasets = {'breast_cancer': {}, 'iris': {}, 'wine': {}}
local_datasets = {'diabetes': {}, 'abalone': {}, 'thoracic': {'min_depth': 3},
                  'bank_marketing': {}, 'credit_card': {}, 'adult': {}}


# @add_cache_support()
@functools.lru_cache(16)
def get_dataset(data_name, discrete=False, seed=None, split=False,
                train_size=0.75, shuffle=True, one_hot=True, verbose=1):
    if data_name in sklearn_datasets:
        if data_name == 'breast_cancer':
            data = load_breast_cancer()
        elif data_name == 'iris':
            data = load_iris()
        else:  # data_name == 'wine':
            data = load_wine()
        data['is_categorical'] = np.array([False] * data['data'].shape[1])
        opts = sklearn_datasets[data_name]
    elif data_name in local_datasets:
        data = load_data(data_name)
        opts = local_datasets[data_name]

    else:
        raise LookupError("Unknown data_name: {}".format(data_name))

    is_categorical = data['is_categorical']

    x = data['data']
    y = data['target']
    # feature_names = data['feature_names']

    if one_hot:
        if verbose:
            print('Converting categorical features to one hot numeric')
        one_hot_features = is_categorical
        if 'is_binary' in data:  # We don't want to one hot already binary data
            one_hot_features = np.logical_and(is_categorical, np.logical_not(data['is_binary']))
        one_hot_encoder = OneHotEncoder(categorical_features=one_hot_features).fit(data['data'])
        data['one_hot_encoder'] = one_hot_encoder
        if verbose:
            print('Total number of categorical features:', np.sum(one_hot_features))
            if hasattr(one_hot_encoder, 'n_values_'):
                print('One hot value numbers:', one_hot_encoder.n_values_)
    if discrete:
        if verbose:
            print('Discretizing all continuous features using MDLP discretizer')
        discretizer_name = data_name + '-discretizer' + ('' if seed is None else ('-' + str(seed))) + '.pkl'
        discretizer_path = get_path(_cached_path, discretizer_name)
        min_depth = 0 if 'min_depth' not in opts else opts['min_depth']
        discretizer = get_discretizer(x, y, continuous_features=np.logical_not(is_categorical),
                                      filenames=discretizer_path, min_depth=min_depth)
        # data['data'] = discretizer.transform(x)
        data['discretizer'] = discretizer

    if split:
        names = [get_path(_datasets_path, data_name + suffix)
                 for suffix in ['/train_x.npy', '/test_x.npy', '/train_y.npy', '/test_y.npy']
                 ]
        train_x, test_x, train_y, test_y = get_split(x, y, train_size=train_size, shuffle=shuffle, filenames=names)
        data.update({
            'train_x': train_x,
            'test_x': test_x,
            'train_y': train_y,
            'test_y': test_y,
        })

    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    ranges = np.vstack([mins, maxs]).T
    data['ranges'] = ranges
    # hacker for some feature_names are arrays
    for key in ['feature_names', 'target_names']:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    if verbose > 0:
        print("-----------------------")
        print("Data Specs: {:s}".format(data_name))
        print("#data: {:d}".format(len(data['target'])))
        print("#features: {:d}".format(data['data'].shape[1]))
        print("#labels: {:d}".format(len(np.unique(data['target']))))
        print("-----------------------")
    return data


@add_cache_support(4)
def get_split(x, y, **kwargs):
    return split_data(x, y, **kwargs)


def categorical2transactions(x: np.ndarray) -> List[List[str]]:
    """
    Convert a 2D int array into a transaction list:
        [
            ['x0=1'， 'x1=0', ...],
            ...
        ]
    :param x:
    :return:
    """
    assert len(x.shape) == 2

    transactions = []
    for entry in x:
        transactions.append(['x%d=%d' % (i, val) for i, val in enumerate(entry)])

    return transactions


def itemset2feature_categories(itemset: Iterable[str]) -> Tuple[List[int], List[int]]:
    features = []
    categories = []
    for item in itemset:
        idx = item.find('=')
        if idx == '-1':
            raise ValueError("No '=' find in the rule!")
        features.append(int(item[1:idx]))
        categories.append(int(item[(idx + 1):]))
    return features, categories


def transactions2freqitems(transactions_by_labels: List[List], supp=0.05, zmin=1, zmax=3) -> List[tuple]:

    supp = int(supp*100)
    itemsets = set()
    for trans in transactions_by_labels:
        itemset = [tuple(sorted(r[0])) for r in eclat(trans, supp=supp, zmin=zmin, zmax=zmax)]
        itemsets |= set(itemset)

    itemsets = list(itemsets)

    logging.info("Total {:d} itemsets mined".format(len(itemsets)))
    return itemsets


def rule_satisfied(x, features, categories) -> np.ndarray:
    """
    return a logical array representing whether entries in x satisfied the rules denoted by features and categories
    :param x: a categorical 2D array
    :param features: a list of feature indices
    :param categories: a list of categories
    :return:
    """
    satisfied = []
    if features[0] == -1 and len(features) == 1:
        # Default rule, all satisfied
        return np.ones(x.shape[0], dtype=bool)
    for idx, cat in zip(features, categories):
        # Every single condition needs to be satisfied.
        satisfied.append(x[:, idx] == cat)
    return functools.reduce(np.logical_and, satisfied)


def categorical2pysbrl_data(x: np.ndarray, y: np.ndarray, data_name, supp=0.05, zmin=1, zmax=3):

    assert len(y.shape) == 1
    assert y.dtype == np.int
    labels = np.unique(y)
    assert max(labels) + 1 == len(labels)

    x_by_labels = []
    for label in labels:
        x_by_labels.append(x[y == label])
    transactions_by_labels = [categorical2transactions(_x) for _x in x_by_labels]
    itemsets = transactions2freqitems(transactions_by_labels, supp=supp, zmin=zmin, zmax=zmax)
    rules = [itemset2feature_categories(itemset) for itemset in itemsets]
    data_by_rule = []
    for features, categories in rules:
        satisfied = rule_satisfied(x, features, categories)
        data_by_rule.append(satisfied)

    # Write data file
    data_filename = get_path(_datasets_path, data_name+'.data')
    before_save(data_filename)
    with open(data_filename, 'w') as f:
        for itemset, data in zip(itemsets, data_by_rule):
            rule_str = '{' + ','.join(itemset) + '}' + '  '
            f.write(rule_str)
            bit_s = ' '.join(['1' if bit else '0' for bit in data])
            f.write(bit_s)
            f.write('\n')

    # Write label file
    label_filename = get_path(_datasets_path, data_name+'.label')
    before_save(label_filename)
    with open(label_filename, 'w') as f:
        for label in labels:
            f.write('{label=%d} ' % label)
            bits = y == label
            bit_s = ' '.join(['1' if bit else '0' for bit in bits])
            f.write(bit_s)
            f.write('\n')
    return data_filename, label_filename


@add_cache_support(n_files=1)
def get_discretizer(x, y, continuous_features=None, seed=None, min_depth=0) -> MDLP:
    discretizer = MDLP(random_state=seed, min_depth=min_depth)
    if continuous_features is not None:
        if continuous_features.dtype == np.bool:
            continuous_features = np.arange(len(continuous_features))[continuous_features]
    discretizer.fit(x, y, continuous_features)
    return discretizer


def sample_balance(x, y, min_ratio=0.5):
    labels, labels_counts = np.unique(y, return_counts=True)
    data = []
    target = []
    max_count = np.max(labels_counts)
    for label, counts in zip(labels, labels_counts):
        logic = y == label
        _x = x[logic]
        _y = y[logic]
        n_repeat = int(np.ceil(max_count * min_ratio / counts))
        if n_repeat < 1:
            n_repeat = 1
        data.append(np.tile(_x, (n_repeat, 1)))
        target.append(np.tile(_y, (n_repeat,)))
    data = np.vstack(data)
    target = np.hstack(target)
    indices = np.arange(len(target))
    np.random.shuffle(indices)
    return data[indices], target[indices]
