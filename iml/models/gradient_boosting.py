"""
A Simple Neural Net Model
"""

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

from iml.models import SKModelWrapper, Regressor, Classifier, CLASSIFICATION, REGRESSION
from iml.models.preprocess import PreProcessMixin, OneHotProcessor, StandardProcessor


class Bagging(PreProcessMixin, SKModelWrapper, Regressor, Classifier):

    def __init__(self, name='bagging', problem=CLASSIFICATION,
                 neurons=(10,), activation='relu', solver='adam',
                 alpha=0.0001, max_iter=1000,
                 standardize=True, one_hot_encoder: OneHotEncoder=None,
                 **kwargs):
        self.scaler = None

        super(Bagging, self).__init__(problem=problem, name=name)

        if problem == CLASSIFICATION:
            self._model = MLPClassifier(hidden_layer_sizes=neurons, activation=activation,
                                        solver=solver, alpha=alpha,
                                        max_iter=max_iter, **kwargs)
        elif problem == REGRESSION:
            self._model = MLPRegressor(hidden_layer_sizes=neurons, activation=activation,
                                       solver=solver, alpha=alpha,
                                       max_iter=max_iter, **kwargs)
        else:
            raise ValueError("Unrecognized problem type {}".format(problem))

        # self.standardize = standardize
        is_numerical = None
        if one_hot_encoder is not None and one_hot_encoder.categorical_features is not None:
            is_numerical = np.logical_not(one_hot_encoder.categorical_features)
        if standardize:
            self.add_processor(StandardProcessor(is_numerical, transform_y=problem == REGRESSION))
        if one_hot_encoder is not None:
            self.add_processor(OneHotProcessor(one_hot_encoder))

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

    @property
    def type(self):
        return 'bag'
