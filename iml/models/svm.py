"""
Simple Wrapper for SVM
"""

import numpy as np

from sklearn.svm import SVC, SVR
from sklearn.preprocessing import OneHotEncoder

from iml.models import SKModelWrapper, Regressor, Classifier, CLASSIFICATION, REGRESSION
from iml.models.preprocess import PreProcessMixin, OneHotProcessor, StandardProcessor


class SVM(PreProcessMixin, SKModelWrapper, Regressor, Classifier):

    def __init__(self, name='nn', problem=CLASSIFICATION,
                 C=1.0, kernel='rbf', probability=True, tol=1e-5,
                 standardize=True, one_hot_encoder: OneHotEncoder = None,
                 **kwargs):
        self.scaler = None

        super(SVM, self).__init__(problem=problem, name=name)

        if problem == CLASSIFICATION:
            self._model = SVC(C=C, kernel=kernel, probability=probability,
                              tol=tol, **kwargs)
        elif problem == REGRESSION:
            self._model = SVR(C=C, kernel=kernel, tol=tol, **kwargs)
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
    def n_support(self):
        return self.model.n_support_

    def evaluate(self, x, y, stage='train'):
        if self._problem == CLASSIFICATION:
            return Classifier.evaluate(self, x, y, stage=stage)
        elif self._problem == REGRESSION:
            return Regressor.evaluate(self, x, y, stage=stage)
        else:
            raise ValueError("Unrecognized problem type {}".format(self._problem))

    @property
    def type(self):
        return 'svm'
