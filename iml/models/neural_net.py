"""
A Simple Neural Net Model
"""

from sklearn.neural_network import MLPClassifier, MLPRegressor

from iml.models import SKModelWrapper, Regressor, Classifier


class NeuralNet(SKModelWrapper, Regressor, Classifier):

    def __init__(self, name='nn', problem="classification",
                 neurons=(10,), activation='relu', solver='sgd', alpha=0.0001, learning_rate_init=0.001):
        super(NeuralNet, self).__init__(name)
        self.problem = problem
        if problem == 'classification':
            self._model = MLPClassifier(hidden_layer_sizes=neurons, activation=activation,
                                       solver=solver, alpha=alpha, learning_rate='adaptive',
                                       learning_rate_init=learning_rate_init)
        elif problem == 'regression':
            self._model = MLPRegressor(hidden_layer_sizes=neurons, activation=activation,
                                      solver=solver, alpha=alpha, learning_rate='adaptive',
                                      learning_rate_init=learning_rate_init)
        else:
            raise ValueError(f"Unrecognized problem type {problem}")

    @property
    def type(self):
        if self.problem == "classification":
            return 'nn-classifier'
        elif self.problem == 'regressor':
            return 'nn-regressor'
        return 'nn'

    @property
    def model(self):
        return self._model