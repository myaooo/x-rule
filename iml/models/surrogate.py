from typing import Callable, List, Optional, Union, Any

import numpy as np

from iml.models import ModelBase


SampleFn = Callable[[], np.ndarray]
ArrayLike = Union[List[Union[np.ndarray,float]], np.ndarray]


class Mixture:
    def __init__(self, sample_fns: List[SampleFn], weights: Optional[np.ndarray] = None):
        self.sample_fns = sample_fns
        self.weights = np.ones(len(sample_fns))/len(sample_fns) if weights is None else weights

    def sample(self, n: int) -> List[np.ndarray]:
        results = []
        n_samples = np.random.choice(len(self), size=n, p=self.weights)
        for i in n_samples:
            results.append(self.sample_fns[i]())
        return results

    def __len__(self) -> int:
        return len(self.sample_fns)


def sigma2cov(sigmas: Union[int, List[int], np.ndarray], n: Optional[int]) -> np.ndarray:
    if isinstance(sigmas, int):
        assert isinstance(n, int)
        return np.eye(n)*sigmas
    else:
        sigmas = np.array(sigmas)
    if len(sigmas.shape) == 1:
        return np.diag(sigmas)
    return sigmas


def gaussian_mixture(means: ArrayLike,
                     covs: ArrayLike,
                     weights: Optional[np.ndarray] = None
                     ) -> Mixture:
    sample_fns = [
        lambda: np.random.multivariate_normal(mean, sigma2cov(cov, len(mean)))
        for mean, cov in zip(means, covs)
    ]
    return Mixture(sample_fns, weights)


def gaussian_mixture_sample(n: int,
                            means: ArrayLike,
                            covs: ArrayLike,
                            weights: Optional[np.ndarray] = None
                            ) -> List[np.ndarray]:
    return gaussian_mixture(means, covs, weights).sample(n)


class Surrogate(ModelBase):

    def __init__(self, name: str):
        super(Surrogate, self).__init__(name)
        self.target = None  # type: Optional[ModelBase]
        self.data_distribution = None

    def surrogate(self, target: ModelBase, instances: np.ndarray, sigma: float, n_sampling: int):
        # if n_sampling is None:
        #     n_sampling = int(self.sampling_rate * len(instances))
        self.target = target
        self.data_distribution = gaussian_mixture(instances, [sigma] * len(instances))
        train_x = self.data_distribution.sample(n_sampling)
        train_y = target.infer(train_x)
        self.train(train_x, train_y)

    def fidelity(self, x):
        if self.target is None:
            raise RuntimeError("The target model has to be set before calling this method!")
        target_y = self.target.infer(x)
        predict_y = self.infer(x)


    @property
    def type(self):
        return 'surrogate'

    def train(self, x, y):
        raise NotImplementedError("Base class")

    def test(self, x, y):
        raise NotImplementedError("Base class")

    def infer(self, x):
        raise NotImplementedError("Base class")