import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from vendors.mdlpc import MDLPDiscretizer
# from iml.models import Tree, SBRL
from iml.data_processing import split

def prep_data():
    data = load_breast_cancer()
    x = data['data']
    y = data['target']
    train_data, test_data = split(list(zip(x,y)),[0.8,0.2],shuffle=True)
    train_x, train_y = zip(*train_data)
    test_x, test_y = zip(*test_data)
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def descritize(x, y, data_name):
    df = pd.DataFrame(x)
    df['label']=y
    descritizer = MDLPDiscretizer(df, 'label')
    descritizer.save(data_name+'.csv', data_name+'.info')

def train_tree(name='tree'):
    train_x, train_y, test_x, test_y = prep_data()
    tree = Tree(name, max_depth=5, min_samples_leaf=3)
    tree.train(train_x, train_y)
    tree.test(test_x, test_y)
    tree.describe()
    tree.export(f'{name}.json')
    tree.save()

def train_rule(name='tree'):
    train_x, train_y, test_x, test_y = prep_data()
    print(train_x.shape, train_x.dtype)
    brl = SBRL()
    brl.fit(train_x, train_y)
    print(brl.predict(test_x))
    brl.print_model()


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = prep_data()
    descritize(train_x, train_y, 'breast_cancer')