import pickle
from typing import Optional, Union

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error, r2_score

FILE_EXTENSION = 'mdl'


def format_name(name):
    return f"{name}.{FILE_EXTENSION}"


# class Metrics:
#     @staticmethod
#     def accuracy(y_true, y_predict):
#         return accuracy_score(y_true, y_predict)
#
#     @staticmethod
#     def log_loss(y_true, y_predict):
#         return log_loss(y_true, y_predict)


class ModelBase:
    def __init__(self, name):
        self.name = name

    @property
    def type(self):
        raise NotImplementedError("Base class")

    def train(self, x, y):
        raise NotImplementedError("Base class")

    def test(self, x, y):
        raise NotImplementedError("Base class")

    def infer(self, x):
        raise NotImplementedError("Base class")

    def save(self, filename=None):
        if filename is None:
            filename = format_name(self.name)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            mdl = pickle.load(f)
            if isinstance(mdl, cls):
                return mdl
            else:
                raise RuntimeError("The loaded file is not a Tree model!")


class SKModelWrapper(ModelBase):
    """A wrapper that wraps models in Sklearn"""
    def __init__(self, name):
        super(SKModelWrapper, self).__init__(name)
        self._model = None  # type: Optional[Union[RegressorMixin, ClassifierMixin]]

    @property
    def type(self):
        return "sk-model-wrapper"

    @property
    def model(self):
        raise NotImplementedError("This is the SKModelWrapper base class!")

    def train(self, x, y):
        self.model.fit(x, y)
        s = self.model.score(x,y)
        print(f"training score: {s}")

    def test(self, x, y):
        """

        :param x:
        :param y:
        :return: accuracy
        """
        s = self.model.score(x, y)
        print(f"testing score: {s}")
        return s

    def infer(self, x):
        return self.model.predict(x)


class Classifier(ModelBase):

    @property
    def type(self):
        return 'classifier'

    def train(self, x, y):
        raise NotImplementedError("This is the classifier base class")

    def test(self, x, y):
        raise NotImplementedError("This is the classifier base class")

    def infer(self, x):
        raise NotImplementedError("This is the classifier base class")

    def log_loss(self, x, y):
        y_pred = self.infer(x)
        return log_loss(y, y_pred)

    def accuracy(self, x, y):
        y_pred = self.infer(x)
        return accuracy_score(y, y_pred)


class Regressor(ModelBase):
    @property
    def type(self):
        return 'classifier'

    def train(self, x, y):
        raise NotImplementedError("This is the classifier base class")

    def test(self, x, y):
        raise NotImplementedError("This is the classifier base class")

    def infer(self, x):
        raise NotImplementedError("This is the classifier base class")

    def mse(self, x, y):
        y_pred = self.infer(x)
        return mean_squared_error(y, y_pred)

    def r2(self, x, y):
        y_pred = self.infer(x)
        return r2_score(y, y_pred)