import pickle

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

from iml.utils.io_utils import obj2json
from iml.models import SKModelWrapper, Classifier, Regressor
from iml.models.surrogate import Surrogate


class Tree(SKModelWrapper, Classifier, Regressor):
    """
    A wrapper class that wraps sklearn.tree.DecisionTreeClassifier
    """
    def __init__(self, problem='classification', name='tree', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1):
        super(Tree, self).__init__(name)
        self.problem = problem
        if problem == 'classification':
            self._model = DecisionTreeClassifier(max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf)
        elif problem == 'regression':
            self._model = DecisionTreeRegressor(max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf)
        else:
            raise ValueError(f"Unrecognized problem type {problem}")

    @property
    def type(self):
        if self.problem == "classification":
            return 'tree-classifier'
        elif self.problem == 'regressor':
            return 'tree-regressor'
        return 'tree'

    @property
    def model(self):
        return self._model

    def describe(self):
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
            return obj2json(nodes, filename)
        raise ValueError(f'Unsupported value "{type}" for argument "type"')


def load(filename):
    return Tree.load(filename)


class TreeSurrogate(Tree, Surrogate):
    def __init__(self, problem='classification', name='tree', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1):
        Surrogate.__init__(self, name)
        Tree.__init__(self, problem, name, max_depth, min_samples_split, min_samples_leaf)