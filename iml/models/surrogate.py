from typing import Callable, List, Optional, Union, Any

import numpy as np

from iml.models import ModelBase, ModelInterface


SampleFn = Callable[[], np.ndarray]
ArrayLike = Union[List[Union[np.ndarray,float]], np.ndarray]


class GaussianDistribution:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, n: Optional[int]=None):
        return np.random.multivariate_normal(self.mean, self.cov, n)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


class Mixture:
    def __init__(self, sample_fns: List[SampleFn], weights: Optional[np.ndarray] = None):
        self.sample_fns = sample_fns
        self.weights = np.ones(len(sample_fns))/len(sample_fns) if weights is None else weights

    def sample(self, n: int) -> List[np.ndarray]:
        results = []
        n_samples = np.random.choice(len(self), size=n, p=self.weights)
        # print('n_samples:')
        # print(n_samples)
        for i in n_samples.tolist():
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
        GaussianDistribution(mean, sigma2cov(cov, len(mean))) for mean, cov in zip(means, covs)
    ]
    # print(sample_fns[0].cov)
    return Mixture(sample_fns, weights)


def gaussian_mixture_sample(n: int,
                            means: ArrayLike,
                            covs: ArrayLike,
                            weights: Optional[np.ndarray] = None
                            ) -> List[np.ndarray]:
    return gaussian_mixture(means, covs, weights).sample(n)


class SurrogateMixin(ModelBase):

    def __init__(self, name: str):
        super(SurrogateMixin, self).__init__(name)
        self.target = None  # type: Optional[ModelBase]
        self.data_distribution = None

    def surrogate(self, target: ModelInterface, instances: np.ndarray,
                  sigmas: Union[List[float], np.ndarray], n_sampling: int,
                  **kwargs):
        # if n_sampling is None:
        #     n_sampling = int(self.sampling_rate * len(instances))
        self.target = target
        self.data_distribution = gaussian_mixture(instances, [sigmas]*len(instances))
        train_x = self.data_distribution.sample(n_sampling)
        train_y = target.predict(train_x)
        self.train(train_x, train_y, **kwargs)

    def fidelity(self, x):
        if self.target is None:
            raise RuntimeError("The target model has to be set before calling this method!")
        y_target = self.target.predict(x)
        y_pred = self.predict(x)

        return self.score(y_target, y_pred)

    def self_test(self, n_sample=200):
        x = self.data_distribution.sample(n_sample)
        fidelity = self.fidelity(x)
        print("Self test fidelity: {:.5f}".format(fidelity))

    def evaluate(self, x, y, stage='train'):
        prefix = 'Training'
        if stage == 'test':
            prefix = 'Testing'
        y_pred = self.predict(x)
        fidelity = self.fidelity(x)
        score = self.score(y, y_pred)
        print(prefix + " fidelity: {:.5f}; score: {:.5f}".format(fidelity, score))

    @property
    def type(self):
        return 'surrogate'
