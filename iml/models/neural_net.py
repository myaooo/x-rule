"""
A Simple Neural Net Model
"""
from typing import Union

import numpy as np

from numpy.random import RandomState

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from iml.models import SKModelWrapper, Regressor, Classifier, CLASSIFICATION, REGRESSION


class NeuralNet(SKModelWrapper, Regressor, Classifier):

    def __init__(self, name='nn', problem=CLASSIFICATION,
                 neurons=(10,), activation='relu', solver='adam',
                 alpha=0.0001, max_iter=1000, standardize=True,
                 one_hot_encoder=None, **kwargs):
        super(NeuralNet, self).__init__(problem=problem, name=name)
        self.scaler = None
        self._model = None  # type: Union[MLPClassifier, MLPRegressor]
        self.is_categorical = None
        self.one_hot_encoder = one_hot_encoder  # type: OneHotEncoder
        if one_hot_encoder is not None:
            self.is_categorical = one_hot_encoder.categorical_features
        if standardize:
            self.scaler = (StandardScaler(), StandardScaler())
        if self._problem == CLASSIFICATION:
            self._model = MLPClassifier(hidden_layer_sizes=neurons, activation=activation,
                                        solver=solver, alpha=alpha,
                                        max_iter=max_iter, **kwargs)
        elif problem == REGRESSION:
            self._model = MLPRegressor(hidden_layer_sizes=neurons, activation=activation,
                                       solver=solver, alpha=alpha,
                                       max_iter=max_iter, **kwargs)
        else:
            raise ValueError("Unrecognized problem type {}".format(problem))

    def fit_transformer(self, x, y):
        if self.is_categorical is None:
            self.is_categorical = [False] * x.shape[1]
        is_categorical = self.is_categorical
        if self.scaler is not None:
            is_numerical = np.logical_not(is_categorical)
            scale_x = x[:, is_numerical]
            self.scaler[0].fit(scale_x)
            if self._problem == REGRESSION:
                self.scaler[1].fit(y)

    def transform_data(self, x, y=None):
        _x, _y = x.copy(), y

        if self.scaler is not None:
            is_numerical = np.logical_not(self.is_categorical)
            _scale_x = self.scaler[0].transform(_x[:, is_numerical])
            _x[:, is_numerical] = _scale_x
            if self._problem == REGRESSION and y is not None:
                _y = self.scaler[1].transform(y)
        if self.one_hot_encoder:
            _x = self.one_hot_encoder.transform(_x)
        if y is None:
            return _x
        return _x, _y

    @property
    def neurons(self):
        return self.model.hidden_layer_sizes

    @property
    def activation(self):
        return self.model.activation

    def evaluate(self, x, y, stage='train'):
        if self._problem == CLASSIFICATION:
            return Classifier.evaluate(self, x, y, stage=stage)
        elif self._problem == REGRESSION:
            return Regressor.evaluate(self, x, y, stage=stage)
        else:
            raise ValueError("Unrecognized problem type {}".format(self._problem))

    def train(self, x, y, feature_names=None, label_names=None, **kwargs):
        self.fit_transformer(x, y)
        _x, _y = self.transform_data(x, y)
        self.model.fit(_x, _y)
        self.evaluate(x, y, stage='train')

    def predict_prob(self, x):
        assert self._problem == CLASSIFICATION
        _x = self.transform_data(x)
        return self.model.predict_proba(_x)

    def predict(self, x):
        _x = self.transform_data(x)
        y = self.model.predict(_x)
        if self._problem == REGRESSION and self.scaler is not None:
            y = self.scaler[1].inverse_transform(y)
        return y


    # def score(self, y_true, y_pred):
    #     if self._problem == CLASSIFICATION:
    #         return self.accuracy(y_true, y_pred)
    #     elif self._problem == REGRESSION:
    #         return self.mse(y_true, y_pred)
    #     else:
    #         raise RuntimeError("Unknown problem type: {:}".format(self._problem))

    @property
    def type(self):
        # if self._problem == CLASSIFICATION:
        #     return 'nn-classifier'
        # elif self._problem == REGRESSION:
        #     return 'nn-regressor'
        return 'nn'

    @property
    def model(self):
        return self._model
