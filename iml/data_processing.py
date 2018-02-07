from typing import List, Union, Tuple, Set, Iterable
import logging
import functools

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from mdlp.discretization import MDLP
from fim import fpgrowth, eclat

from iml.utils.io_utils import before_save, get_path, file_exists, load_file, save_file


_datasets_path = get_path('datasets')
_cached_path = get_path('datasets/cached')

# print(_datasets_path)

sklearn_datasets = {'breast_cancer', 'iris', 'wine'}

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


# @add_cache_support()
@functools.lru_cache(16)
def get_dataset(data_name, discrete=False, seed=None, split=None, train_size=0.75, shuffle=True, verbose=1):
    data = None
    if data_name in sklearn_datasets:
        if data_name == 'breast_cancer':
            data = load_breast_cancer()
        elif data_name == 'iris':
            data = load_iris()
        elif data_name == 'wine':
            data = load_wine()
    else:
        raise LookupError("Unknown data_name: {}".format(data_name))
    x = data['data']
    y = data['target']
    # feature_names = data['feature_names']
    if discrete:
        discretizer_name = data_name + '-discretizer' + ('' if seed is None else ('-' + str(seed))) + '.pkl'
        discretizer_path = get_path(_cached_path, discretizer_name)
        discretizer = get_discretizer(x, y, filenames=discretizer_path)
        data['data'] = discretizer.transform(x)
        data['discretizer'] = discretizer
    if split:
        names = [get_path(_datasets_path, data_name + suffix)
                 for suffix in ['/train_x.npy', '/train_y.npy', '/test_x.npy', '/test_y.npy']
                 ]
        train_x, test_x, train_y, test_y = get_split(x, y, train_size=train_size, shuffle=shuffle, filenames=names)
        data.update({
            'train_x': train_x,
            'test_x': test_x,
            'train_y': train_y,
            'test_y': test_y,
        })
    for key in ['feature_names', 'target_names']:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    if verbose > 0:
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
def get_discretizer(x, y, seed=None) -> MDLP:
    discretizer = MDLP(random_state=seed)
    discretizer.fit(x, y)
    return discretizer
