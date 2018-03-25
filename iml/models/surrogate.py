from typing import Callable, List, Optional, Union
from collections import defaultdict

import numpy as np
from numpy.random import multivariate_normal
from scipy import stats

from iml.models import ModelBase, ModelInterface
from iml.utils.io_utils import save_file, get_path, load_file


sample_cache_dir = 'models/sample_cache/'
ArrayLike = Union[List[Union[np.ndarray, float]], np.ndarray]
#
#
# class GaussianDistribution:
#     def __init__(self, mean, cov):
#         self.mean = mean
#         self.cov = cov
#
#     def sample(self, n: Optional[int]=None):
#         return np.random.multivariate_normal(self.mean, self.cov, n)
#
#     def __call__(self, *args, **kwargs):
#         return self.sample(*args, **kwargs)
#
#
# class Mixture:
#     def __init__(self, sample_fns: List[SampleFn], weights: Optional[np.ndarray] = None):
#         self.sample_fns = sample_fns
#         self.weights = np.ones(len(sample_fns))/len(sample_fns) if weights is None else weights
#
#     def sample(self, n: int) -> List[np.ndarray]:
#         results = []
#         n_samples = np.random.choice(len(self), size=n, p=self.weights)
#         # print('n_samples:')
#         # print(n_samples)
#         for i in n_samples.tolist():
#             results.append(self.sample_fns[i]())
#         return np.array(results)
#
#     def __len__(self) -> int:
#         return len(self.sample_fns)


def sigma2cov(sigmas: Union[int, List[int], np.ndarray], n: Optional[int]) -> np.ndarray:
    if isinstance(sigmas, int):
        assert isinstance(n, int)
        return np.eye(n)*sigmas
    else:
        sigmas = np.array(sigmas)
    if len(sigmas.shape) == 1:
        return np.diag(sigmas)
    return sigmas


def gaussian_mixture(means: np.ndarray,
                     cov: np.ndarray,
                     weights: Optional[np.ndarray] = None
                     ) -> Callable[[int], np.ndarray]:
    # sample_fns = [
    #     GaussianDistribution(mean, sigma2cov(cov, len(mean))) for mean, cov in zip(means, covs)
    # ]

    if weights is None:
        weights = np.empty(len(means), dtype=np.float32)
        weights.fill(1/len(means))
    n_features = len(means[0])

    # def sample(n: int) -> np.ndarray:
    #     results = []
    #     if n < len(means):
    #         n_samples = np.random.choice(len(means), size=n, p=weights)
    #         for i in n_samples.tolist():
    #             results.append(multivariate_normal(means[i], sigma2cov(covs[i], n_features)))
    #         return np.array(results)
    #     else:
    #         n_samples = np.random.multinomial(n, weights).reshape(-1)
    #         for idx, num in enumerate(n_samples):
    #             if num == 0:
    #                 continue
    #             results.append(multivariate_normal(means[idx], sigma2cov(covs[idx], n_features), num))
    #         return np.vstack(results)

    def sample(n: int) -> np.ndarray:
        norm = multivariate_normal(np.zeros((n_features, ), dtype=np.float), cov, n)
        indices = np.random.choice(len(means), size=n, p=weights)
        return norm + means[indices]

    return sample


def scotts_factor(n, d):
    return n ** (-1./(d+4))


INTEGER = 'integer'
CONTINUOUS = 'continuous'
CATEGORICAL = 'categorical'

data_type = {INTEGER, CONTINUOUS, CATEGORICAL}


class IntegerConstraint:
    def __init__(self, _range=None):
        self._range = _range

    def regularize(self, arr: np.ndarray):
        assert len(arr.shape) == 1
        if self._range is not None:
            arr[arr > self._range[1]] = self._range[1]
            arr[arr < self._range[0]] = self._range[0]
        arr = np.round(arr)
        return arr

    @property
    def type(self):
        return INTEGER


class CategoricalConstraint:
    def __init__(self, categories=None):
        self._categories = categories

    @property
    def type(self):
        return CATEGORICAL


class ContinuousConstraint:
    def __init__(self, _range=None):
        self._range = _range

    def regularize(self, arr: np.ndarray):
        assert len(arr.shape) == 1
        if self._range is not None:
            arr[arr > self._range[1]] = self._range[1]
            arr[arr < self._range[0]] = self._range[0]
        return arr

    @property
    def type(self):
        return CONTINUOUS


def create_constraint(feature_type, **kwargs):
    if feature_type == INTEGER:
        return IntegerConstraint(_range=kwargs['_range'])
    elif feature_type == CONTINUOUS:
        return ContinuousConstraint(_range=kwargs['_range'])
    elif feature_type == CATEGORICAL:
        return CategoricalConstraint()
    else:
        raise ValueError("Unknown feature_type {}".format(feature_type))


def create_constraints(is_categorical: np.ndarray, is_continuous: np.ndarray, ranges):
    constraints = []
    for i in range(len(is_categorical)):
        feature_type = CATEGORICAL if is_categorical[i] else CONTINUOUS if is_continuous[i] else INTEGER
        constraints.append(create_constraint(feature_type, _range=ranges[i]))
    return constraints


def create_sampler(instances: np.ndarray, constraints, cov_factor=1.0, verbose=False) -> Callable[[int], np.ndarray]:
    """
    We treat the sampling of categorical values as a multivariate categorical distribution.
    We sample categorical values first, then sample the continuous and integer variables
    using the conditional distribution w.r.t. to the categorical vector.

    Note: the category features only support at most 128 number of choices

    :param instances:
    :param constraints:
    :param verbose: a flag for debugging output
    :return:
    """
    is_categorical = [True if constraint.type == CATEGORICAL else False for constraint in constraints]
    is_integer = [True if constraint.type == INTEGER else False for constraint in constraints]
    is_continuous = [True if constraint.type == CONTINUOUS else False for constraint in constraints]
    is_numeric = np.logical_or(is_integer, is_continuous)
    # sigmas = np.array([constraint.sigma for constraint in constraints if constraint.type != CATEGORICAL])

    n_features = len(is_categorical)
    n_samples = len(instances)

    def _build_cache():
        categoricals = instances[:, is_categorical].astype(np.int8)

        categorical_samples = defaultdict(list)
        for i in range(n_samples):
            key = bytes(categoricals[i, :])
            categorical_samples[key].append(instances[i])

        keys = []
        probs = []
        key2instances = []
        for key, value in categorical_samples.items():
            keys.append(key)
            probs.append(len(value) / n_samples)
            key2instances.append(np.array(value))
        if verbose:
            print("# of categories:", len(keys))
            print("Distribution of # of instances per categories:")
            hists, bins = np.histogram(probs * n_samples, 5)
            print("hists:", hists.tolist())
            print("bins:", bins.tolist())
        return keys, probs, key2instances
    cat_keys, cat_probs, cat2instances = _build_cache()

    # Try stats.gaussian_kde
    glb_kde = stats.gaussian_kde(instances[:, is_numeric].T, 'silverman')
    cov = cov_factor * glb_kde.covariance

    def sample(n: int) -> np.ndarray:
        samples = []
        sample_nums = np.random.multinomial(n, cat_probs)
        for idx, num in enumerate(sample_nums):
            if num == 0:
                continue
            sample_buffer = np.empty((num, n_features), dtype=np.float)
            sample_buffer[:, is_numeric] = gaussian_mixture(cat2instances[idx][:, is_numeric], cov)(num)
            categorical_part = np.frombuffer(cat_keys[idx], dtype=np.int8)
            sample_buffer[:, is_categorical] = np.tile(categorical_part, (num, 1)).astype(np.float)

            samples.append(sample_buffer)
        sample_mat = np.vstack(samples)

        # regularize integer part
        for i, constraint in enumerate(constraints):
            if constraint.type == INTEGER:
                sample_mat[:, i] = constraint.regularize(sample_mat[:, i])

        return sample_mat

    return sample


class SurrogateMixin(ModelBase):

    def __init__(self, **kwargs):
        super(SurrogateMixin, self).__init__(**kwargs)
        self.target = None  # type: Optional[ModelBase]
        self.data_distribution = None
        self._n_samples = None
        self.train_fidelity = None
        self.test_fidelity = None
        self.self_test_fidelity = None

    def surrogate(self, target: ModelInterface, instances: np.ndarray,
                  constraints: list, sampling_rate: float=5., cache=True, cov_factor: float=1.0,
                  **kwargs):
        n_samples = int(sampling_rate * len(instances))

        self.target = target
        self.data_distribution = create_sampler(instances, constraints, cov_factor)
        train_x = self.data_distribution(n_samples)
        train_y = target.predict(train_x).astype(np.int)
        self.train(train_x, train_y, **kwargs)
        self.evaluate(train_x, train_y, stage='train')
        self.self_test(int(n_samples * 0.2), cache=cache)
        if cache:
            self.cache_sample(train_x, is_train=True)

    def cache_sample(self, x, is_train=False):
        file_name = self.name + ('-train' if is_train else '-test') + '.csv'
        file_path = get_path(sample_cache_dir, file_name)
        save_file(x, file_path)

    def load_cache(self, is_train=False):
        file_name = self.name + ('-train' if is_train else '-test') + '.csv'
        file_path = get_path(sample_cache_dir, file_name)
        return load_file(file_path)

    def sample(self, n: int):
        assert self.data_distribution is not None
        return self.data_distribution(n)

    def fidelity(self, x):
        if self.target is None:
            raise RuntimeError("The target model has to be set before calling this method!")
        y_target = self.target.predict(x)
        y_pred = self.predict(x)

        return self.score(y_target, y_pred)

    def self_test(self, n_sample=200, cache=True):
        x = self.data_distribution(n_sample)
        fidelity = self.fidelity(x)
        print("Self test fidelity: {:.5f}".format(fidelity))
        self.self_test_fidelity = fidelity
        if cache:
            self.cache_sample(x, is_train=False)
        return fidelity

    def evaluate(self, x, y, stage='train'):
        prefix = 'Training'

        y_pred = self.predict(x)
        fidelity = self.fidelity(x)
        score = self.score(y, y_pred)
        if stage == 'test':
            prefix = 'Testing'
            self.train_fidelity = fidelity
        else:
            self.test_fidelity = fidelity
        print(prefix + " fidelity: {:.5f}; score: {:.5f}".format(fidelity, score))
        return fidelity, score

    @property
    def type(self):
        return 'surrogate'
