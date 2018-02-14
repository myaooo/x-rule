import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

from iml.utils.io_utils import dict2json
from iml.models import SKModelWrapper, Classifier, Regressor, CLASSIFICATION, REGRESSION
from iml.models.surrogate import SurrogateMixin
from iml.models.preprocess import PreProcessMixin, OneHotProcessor, StandardProcessor


class Tree(PreProcessMixin, SKModelWrapper, Classifier, Regressor):
    """
    A wrapper class that wraps sklearn.tree.DecisionTreeClassifier
    """

    def __init__(self, problem=CLASSIFICATION, name='tree',
                 criterion='gini', splitter='best',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 one_hot_encoder=None, **kwargs):
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
            raise ValueError("Unrecognized problem type {}".format(problem))

        if one_hot_encoder is not None:
            self.add_processor(OneHotProcessor(one_hot_encoder))

    @property
    def n_classes(self):
        return self.model.tree_.max_n_classes

    @property
    def n_features(self):
        return self.model.tree_.n_features

    @property
    def n_nodes(self):
        return self.model.tree_.node_count

    @property
    def max_depth(self):
        return self.model.tree_.max_depth

    @property
    def type(self):
        if self._problem == CLASSIFICATION:
            return 'tree-classifier'
        elif self._problem == REGRESSION:
            return 'tree-regressor'
        return 'tree'

    def describe(self, feature_names=None):
        tree = self.model.tree_
        print("Depth: {}".format(tree.max_depth))
        print("#Node: {}".format(tree.node_count))

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
            nodes = self.to_dict()
            return dict2json(nodes, filename)
        raise ValueError('Unsupported value "{}" for argument "type"'.format(filetype))

    def to_dict(self):
        tree = self.model.tree_
        children_left, children_right, feature, threshold, impurity, value = \
            [a.tolist() for a in [tree.children_left, tree.children_right,
                                  tree.feature, tree.threshold,
                                  tree.impurity, tree.value]]

        def _build(idx):
            support = value[idx][0]
            node = {
                'value': support,
                'impurity': impurity[idx],
                'idx': idx,
                'output': int(np.argmax(support))
            }
            if children_left[idx] != children_right[idx]:
                # if not leave
                node['feature'] = feature[idx]
                node['threshold'] = threshold[idx]
                node['left'] = _build(children_left[idx])
                node['right'] = _build(children_right[idx])

            return node

        return _build(0)


def load(filename):
    return Tree.load(filename)


class TreeSurrogate(Tree, SurrogateMixin):
    def __init__(self, **kwargs):
        # print(kwargs)
        super(TreeSurrogate, self).__init__(**kwargs)
