"""
A Simple Neural Net Model
"""
from typing import Union

import numpy as np

from numpy.random import RandomState

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from iml.models import SKModelWrapper, Regressor, Classifier, CLASSIFICATION, REGRESSION


class NeuralNet(SKModelWrapper, Regressor, Classifier):

    def __init__(self, name='nn', problem=CLASSIFICATION,
                 neurons=(10,), activation='relu', solver='adam',
                 alpha=0.0001, learning_rate='adaptive', learning_rate_init=0.001,
                 max_iter=1000, tol=1e-5, standardize=True, **kwargs):
        super(NeuralNet, self).__init__(problem=problem, name=name)
        self.scaler = None
        self._model = None  # type: Union[MLPClassifier, MLPRegressor]
        # self._feature_names = None
        # self._label_names = None
        # self.neurons = neurons
        # self.activation = activation
        if standardize:
            self.scaler = (StandardScaler(), StandardScaler())
        if self._problem == CLASSIFICATION:
            self._model = MLPClassifier(hidden_layer_sizes=neurons, activation=activation,
                                        solver=solver, alpha=alpha, learning_rate=learning_rate,
                                        learning_rate_init=learning_rate_init,
                                        max_iter=max_iter, tol=tol, **kwargs)
        elif problem == REGRESSION:
            self._model = MLPRegressor(hidden_layer_sizes=neurons, activation=activation,
                                       solver=solver, alpha=alpha, learning_rate=learning_rate,
                                       learning_rate_init=learning_rate_init,
                                       max_iter=max_iter, tol=tol, **kwargs)
        else:
            raise ValueError("Unrecognized problem type {}".format(problem))

    # @property
    # def feature_names(self):
    #     return self._feature_names
    #
    # @property
    # def label_names(self):
    #     return self._label_names

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
        _x, _y = x, y
        # if feature_names is not None:
        #     self._feature_names = feature_names
        # if label_names is not None:
        #     self._label_names = label_names
        if self.scaler is not None:
            self.scaler[0].fit(x)
            _x = self.scaler[0].transform(x)
            if self._problem == REGRESSION:
                self.scaler[1].fit(y)
                _y = self.scaler[1].transform(y)
        self.model.fit(_x, _y)
        self.evaluate(x, y, stage='train')

    def predict_prob(self, x):
        assert self._problem == CLASSIFICATION
        if self.scaler is not None:
            x = self.scaler[0].transform(x)

        return self.model.predict_proba(x)

    def predict(self, x):
        if self.scaler is not None:
            x = self.scaler[0].transform(x)
        y = self.model.predict(x)
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
