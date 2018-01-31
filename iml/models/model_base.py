import dill as pickle
from typing import Optional, Union

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error, r2_score

from iml.utils.io_utils import get_path, before_save, obj2pkl, pkl2obj
from iml.config import model_dir

FILE_EXTENSION = '.mdl'

CLASSIFICATION = 'classification'
REGRESSION = 'regression'


def _format_name(name):
    return get_path(model_dir(), "{}{}".format(name, FILE_EXTENSION))


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

    def train(self, x, y, **kwargs):
        raise NotImplementedError("Base class")

    def test(self, x, y):
        """

        :param x:
        :param y:
        :return: accuracy
        """
        return self.evaluate(x, y, stage='test')

    def evaluate(self, x, y, stage='train'):
        raise NotImplementedError("Base class")

    # def predict_prob(self, x):
    #     raise NotImplementedError("Base class")

    def predict(self, x):
        raise NotImplementedError("Base class")

    def score(self, y_true, y_pred):
        raise NotImplementedError("Base class")

    def save(self, filename=None):
        if filename is None:
            filename = _format_name(self.name)
        obj2pkl(self, filename)

    @classmethod
    def load(cls, filename):
        mdl = load_model(filename)
        if isinstance(mdl, cls):
            return mdl
        else:
            raise RuntimeError("The loaded file is not a Tree model!")


def load_model(filename: str) -> ModelBase:
    with open(filename, "rb") as f:
        mdl = pickle.load(f)
        # assert isinstance(mdl, ModelBase)
        return mdl


class SKModelWrapper(ModelBase):
    """A wrapper that wraps models in Sklearn"""
    def __init__(self, problem=CLASSIFICATION, name='wrapper'):
        super(SKModelWrapper, self).__init__(name=name)
        self._problem = problem
        self._model = None  # type: Optional[Union[RegressorMixin, ClassifierMixin]]

    @property
    def type(self):
        return "sk-model-wrapper"

    @property
    def model(self):
        raise NotImplementedError("This is the SKModelWrapper base class!")

    def train(self, x, y, **kwargs):
        self.model.fit(x, y)
        self.evaluate(x, y, stage='train')

    def predict_prob(self, x):
        assert self._problem == CLASSIFICATION
        return self.model.predict_proba(x)

    def predict(self, x):
        return self.model.predict(x)

    # def score(self, y_true, y_pred):
    #     raise NotImplementedError("This is the SKModelWrapper base class!")


class Classifier(ModelBase):

    @property
    def type(self):
        return 'classifier'

    # def train(self, x, y):
    #     raise NotImplementedError("This is the classifier base class")

    def evaluate(self, x, y, stage='train'):
        acc = self.accuracy(y, self.predict(x))
        loss = self.log_loss(y, self.predict_prob(x))
        prefix = 'Training'
        if stage == 'test':
            prefix = 'Testing'
        print(prefix + " accuracy: {:.5f}; loss: {:.5f}".format(acc, loss))
        return acc, loss

    def predict_prob(self, x):
        raise NotImplementedError("This is the classifier base class!")

    def score(self, y_true, y_pred):
        return self.accuracy(y_true, y_pred)

    # def infer(self, x):
    #     """
    #     Infer the probability of each classes
    #     :param x: a 2-D array, with shape (n_instances, n_features)
    #     :return: a 2-D array with shape (n_instances, n_classes), representing the probability
    #     """
    #     raise NotImplementedError("This is the classifier base class")
    #
    # def predict(self, x):
    #     """
    #     Predict the class of the instances
    #     :param x: a 2-D array, with shape (n_instances, n_features)
    #     :return: a 1-D array with shape (n_instances,), representing the classes of the prediction
    #     """
    #     raise NotImplementedError("This is the classifier base class")

    @staticmethod
    def log_loss(y_true, y_prob):
        # print(y_true.max())
        return log_loss(y_true, y_prob, labels=list(range(y_prob.shape[1])))

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)


class Regressor(ModelBase):

    @property
    def type(self):
        return 'regressor'

    def evaluate(self, x, y, stage='train'):
        """

        :param x:
        :param y:
        :return: accuracy
        """
        s = self.mse(y, self.predict(x))
        prefix = 'Training'
        if stage == 'test':
            prefix = 'Testing'
        print(prefix + " mse: {:.5f}".format(s))
        return s

    def score(self, y_true, y_pred):
        return self.mse(y_true, y_pred)

    # def predict(self, x):
    #     return self.infer(x)

    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)