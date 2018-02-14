from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from iml.models import ModelBase

from mdlp.discretization import MDLP

class PreProcessBase:

    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    def transform(self, x: np.ndarray, y: np.ndarray=None):
        if y is None:
            return x
        return x, y

    def inverse_transform(self, y):
        return y


class StandardProcessor(PreProcessBase):
    def __init__(self, is_numerical=None, transform_y=False):
        self.is_numerical_ = None
        self.transform_y = transform_y
        self.x_scaler = None  # type: StandardScaler
        self.y_scaler = None  # type: StandardScaler
        if is_numerical is not None:
            assert isinstance(is_numerical, np.ndarray) and is_numerical.dtype == np.bool
            self.is_numerical_ = is_numerical

    def fit(self, x, y=None):
        if self.is_numerical_ is None:
            self.is_numerical_ = np.ones((x.shape[1],), dtype=np.bool)
        scale_x = x[:, self.is_numerical_]
        self.x_scaler = StandardScaler(copy=False).fit(scale_x)  # We manually handle the copy ourselves
        if self.transform_y and y is not None:
            self.y_scaler = StandardScaler(copy=True).fit(y.reshape(-1, 1))
        # super(StandardProcessor, self).fit(x, y)

    def transform(self, x, y=None):
        _x, _y = x, y
        if self.x_scaler is not None:
            _x = _x.copy()
            _x[:, self.is_numerical_] = self.x_scaler.transform(_x[:, self.is_numerical_])
        if self.y_scaler is not None and _y is not None:
            _y = self.y_scaler.transform(_y.reshape(-1, 1)).reshape(-1)
        return _x, _y

    def inverse_transform(self, y):
        if self.y_scaler is not None:
            return self.y_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        return y


class OneHotProcessor(PreProcessBase):
    def __init__(self, encoder: OneHotEncoder=None):
        self.is_categorical_ = None
        self.encoder = encoder
        if encoder is not None:
            if encoder.categorical_features is not None:
                self.is_categorical_ = encoder.categorical_features
            else:
                self.is_categorical_ = np.ones(len(encoder.n_values_), dtype=np.bool)
        super(OneHotProcessor, self).__init__()

    def transform(self, x, y=None):
        if self.encoder is not None:
            x = self.encoder.transform(x)
        return x, y


class DiscreteProcessor(PreProcessBase):
    def __init__(self, discretizer: MDLP):
        self.is_numeical_ = None
        self.discretizer = discretizer
        if discretizer is not None:
            continuous_features = discretizer.continuous_features
            self.is_numeical_ = np.zeros(len(discretizer.cut_points_), dtype=np.bool)
            self.is_numeical_[continuous_features] = True
        super(DiscreteProcessor, self).__init__()

    def fit(self, x: np.ndarray, y: np.ndarray, continuous_features=None):
        if self.discretizer.cut_points_ is not None:
            print("Warning: discretizer has already been fitted. skipped fitting")
        self.discretizer.fit(x, y, continuous_features)
        continuous_features = self.discretizer.continuous_features
        self.is_numeical_ = np.zeros(len(self.discretizer.cut_points_), dtype=np.bool)
        self.is_numeical_[continuous_features] = True

    def transform(self, x, y=None):
        if self.discretizer is not None:
            x = self.discretizer.transform(x)
        return x, y


class PreProcessMixin(ModelBase, PreProcessBase):
    def __init__(self, **kwargs):
        self.processors = []  # type: List[PreProcessBase]
        super(PreProcessMixin, self).__init__(**kwargs)

    def add_processor(self, processor: PreProcessBase):
        self.processors.append(processor)

    def transform(self, x, y=None):
        _x, _y = x, y
        for processor in self.processors:
            _x, _y = processor.transform(_x, _y)
        return super(PreProcessMixin, self).transform(_x, _y)

    def inverse_transform(self, y):
        _y = y
        for processor in reversed(self.processors):
            _y = processor.inverse_transform(_y)
        return _y

    def fit(self, x: np.ndarray, y: np.ndarray=None):
        for processor in self.processors:
            processor.fit(x, y)

    def train(self, x, y, **kwargs):
        # self.fit(x, y)
        self.fit(x, y)
        _x, _y = self.transform(x, y)
        super(PreProcessMixin, self).train(_x, _y)

    def predict_prob(self, x, transform=True, **kwargs):
        if not transform:
            super(PreProcessMixin, self).predict_prob(x, **kwargs)
        _x = self.transform(x)
        return super(PreProcessMixin, self).predict_prob(_x, **kwargs)

    def predict(self, x, transform=True, **kwargs):
        if not transform:
            return super(PreProcessMixin, self).predict(x, **kwargs)
        _x = self.transform(x)
        _y = super(PreProcessMixin, self).predict(_x, **kwargs)
        return self.inverse_transform(_y)
