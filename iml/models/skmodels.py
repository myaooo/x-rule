"""
A wrapper class for sk models
"""

import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OneHotEncoder

from iml.models import Classifier, Regressor
from iml.models.metrics import auc_score
from iml.models.preprocess import PreProcessMixin, OneHotProcessor, StandardProcessor


def _check_methods(obj, methods):
    for method in methods:
        if not hasattr(obj, method):
            raise ValueError('object must have method {}!'.format(method))


class _SKClassifier(Classifier):

    def __init__(self, model: ClassifierMixin, name='sk'):
        super(_SKClassifier, self).__init__(name)
        _check_methods(model, ['predict', 'fit', 'predict_proba'])
        self._model = model
        # self.standardize = standardize

    @property
    def model(self):
        return self._model

    def train(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)
        # return self.evaluate(x, y, stage='train')

    def predict_prob(self, x):
        return self.model.predict_proba(x)

    def predict(self, x):
        return self.model.predict(x)

    @property
    def type(self):
        return 'sk-classifier'


class _SKRegressor(Regressor):

    def __init__(self, model: RegressorMixin, name='sk'):
        super(_SKRegressor, self).__init__(name)
        _check_methods(model, ['predict', 'fit'])
        self._model = model
        # self.standardize = standardize

    @property
    def model(self):
        return self._model

    def train(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)
        # return self.evaluate(x, y, stage='train')

    def predict(self, x):
        return self.model.predict(x)

    @property
    def type(self):
        return 'sk-classifier'


class SKClassifier(PreProcessMixin, _SKClassifier):
    def __init__(self, model: ClassifierMixin, name='sk', standardize=True, one_hot_encoder: OneHotEncoder=None):
        super(SKClassifier, self).__init__(model=model, name=name)
        is_numerical = None
        if one_hot_encoder is not None and one_hot_encoder.categorical_features is not None:
            is_numerical = np.logical_not(one_hot_encoder.categorical_features)
        if standardize:
            self.add_processor(StandardProcessor(is_numerical, transform_y=False))
        if one_hot_encoder is not None:
            self.add_processor(OneHotProcessor(one_hot_encoder))

    # def predict_prob(self, x, transform=True, **kwargs):
    #     return super(SKClassifier, self).predict_prob(x, transform, **kwargs)
    #
    # def predict(self, x, transform=True, **kwargs):
    #     return super(SKClassifier, self).predict(x, transform, **kwargs)

    # def test(self, x, y):
        # raise NotImplementedError("Base class")
        # return self.evaluate(x, y, stage='test')

    def evaluate(self, x, y, stage='train'):

        acc = self.accuracy(y, self.predict(x))
        y_prob = self.predict_prob(x)
        loss = self.log_loss(y, y_prob)
        auc = auc_score(y, y_prob, average='macro')
        prefix = 'Training'
        if stage == 'test':
            prefix = 'Testing'
        print(prefix + " accuracy: {:.5f}; loss: {:.5f}; auc: {:.5f}".format(acc, loss, auc))
        return acc, loss, auc


class SKRegressor(PreProcessMixin, _SKRegressor):
    def __init__(self, model: RegressorMixin, name='sk', standardize=True, one_hot_encoder: OneHotEncoder=None):
        super(SKRegressor, self).__init__(model=model, name=name)
        is_numerical = None
        if one_hot_encoder is not None and one_hot_encoder.categorical_features is not None:
            is_numerical = np.logical_not(one_hot_encoder.categorical_features)
        if standardize:
            self.add_processor(StandardProcessor(is_numerical, transform_y=True))
        if one_hot_encoder is not None:
            self.add_processor(OneHotProcessor(one_hot_encoder))

    # def predict_prob(self, x, transform=True, **kwargs):
    #     return super(SKClassifier, self).predict_prob(x, transform, **kwargs)
    #
    # def predict(self, x, transform=True, **kwargs):
    #     return super(SKClassifier, self).predict(x, transform, **kwargs)

    # def test(self, x, y):
        # raise NotImplementedError("Base class")
        # return self.evaluate(x, y, stage='test')

    def evaluate(self, x, y, stage='train'):

        y_pred = self.predict(x)
        mse = self.mse(y, y_pred)
        # y_prob = self.predict_prob(x)
        # loss = self.log_loss(y, y_prob)
        # auc = auc_score(y, y_prob, average='macro')
        prefix = 'Training'
        if stage == 'test':
            prefix = 'Testing'
        print(prefix + "mse: {:.5f}".format(mse))
        return mse

