import pickle

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

from iml.utils.io_utils import dict2json
from iml.models import SKModelWrapper, Classifier, Regressor, CLASSIFICATION, REGRESSION
from iml.models.surrogate import SurrogateMixin


class Tree(SKModelWrapper, Classifier, Regressor):
    """
    A wrapper class that wraps sklearn.tree.DecisionTreeClassifier
    """

    def __init__(self, problem=CLASSIFICATION, name='tree', criterion='gini', splitter='best',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, **kwargs):
        super(Tree, self).__init__(problem=problem, name=name)
        self._problem = problem
        # self._feature_names = None
        # self._label_names = None
        if problem == CLASSIFICATION:
            self._model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, splitter=splitter,
                                                 min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf, **kwargs)
        elif problem == REGRESSION:
            self._model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, splitter=splitter,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf, **kwargs)
        else:
            raise ValueError(f"Unrecognized problem type {problem}")

    # @property
    # def feature_names(self):
    #     return self._feature_names
    #
    # @property
    # def label_names(self):
    #     return self._label_names

    @property
    def type(self):
        if self._problem == CLASSIFICATION:
            return 'tree-classifier'
        elif self._problem == REGRESSION:
            return 'tree-regressor'
        return 'tree'

    @property
    def model(self):
        return self._model

    def describe(self, feature_names=None):
        tree = self.model.tree_
        print(f"Depth: {tree.max_depth}")
        print(f"#Node: {tree.node_count}")

    def export(self, filename=None, filetype='dot'):
        """

        :param filename: the filename of the export tree, None to return an obj
        :param filetype: 'dot' or 'json'. Default to 'dot
            'dot': use the sklearn.tree.export_graphviz
            'json': export an json string or file (can be used in frontend)
         :return:
        """
        if filename is not None:
            _type = filename.split('.')[-1]
            if _type == 'dot' or _type == 'json':
                filetype = _type
        if filetype == 'dot':
            return export_graphviz(self.model, filename)
        if filetype == 'json':
            tree = self.model.tree_
            children_left, children_right, feature, threshold, impurity, value = \
                [a.tolist() for a in [tree.children_left, tree.children_right,
                                      tree.feature, tree.threshold,
                                      tree.impurity, tree.value]]

            def _build(idx):
                node = {'value': value[idx], 'impurity': impurity[idx]}
                if children_left[idx] != children_right[idx]:
                    # if not leave
                    node['feature'] = feature[idx]
                    node['threshold'] = threshold[idx]
                    node['left'] = _build(children_left[idx])
                    node['right'] = _build(children_right[idx])

                return node

            nodes = _build(0)
            return dict2json(nodes, filename)
        raise ValueError(f'Unsupported value "{type}" for argument "type"')


def load(filename):
    return Tree.load(filename)


class TreeSurrogate(Tree, SurrogateMixin):
    def __init__(self, **kwargs):
        print(kwargs)
        super(TreeSurrogate, self).__init__(**kwargs)
