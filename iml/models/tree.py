import pickle

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

from iml.utils.io_utils import obj2json
from iml.models import ModelBase


class Tree(ModelBase):
    """
    A wrapper class that wraps sklearn.tree.DecisionTreeClassifier
    """
    def __init__(self, problem='classification', name='tree', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1):
        super(Tree, self).__init__(name)
        if problem == 'classification':
            self.model = DecisionTreeClassifier(max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf)
        elif problem == 'regression':
            self.model = DecisionTreeRegressor(max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf)
        else:
            raise ValueError(f"Unrecognized problem type {problem}")

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

    def save(self, filename=None):
        if filename is None:
            filename = f'{self.name}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def infer(self, x):
        return self.model.predict(x)

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

    @property
    def type(self):
        return 'tree'


def describe(tree):
    assert isinstance(tree, DecisionTreeClassifier)
    print(f"Depth: {tree.tree_.max_depth}")


def load(filename):
    return Tree.load(filename)
