from sklearn.datasets import load_breast_cancer

from iml.models.tree import Tree
from iml.data_processing import split


def train_tree(name='tree'):
    data = load_breast_cancer()
    x = data['data']
    y = data['target']
    train_data, test_data = split(list(zip(x,y)),[0.8,0.2],shuffle=True)
    train_x, train_y = zip(*train_data)
    test_x, test_y = zip(*test_data)
    tree = Tree(name, max_depth=5, min_samples_leaf=3)
    tree.train(train_x, train_y)
    tree.test(test_x, test_y)
    tree.describe()
    tree.export(f'{name}.json')
    tree.save()


if __name__ == '__main__':
    train_tree()