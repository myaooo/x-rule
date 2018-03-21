"""
Dataset: Pima Indian Diabetes


"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

from sklearn.preprocessing import StandardScaler

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path


data_path = get_path('datasets/pima', 'diabetes.csv')

header = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
          'Body Mass Index', 'Diabetes Pedigree Function', 'Age', 'Outcome']

is_categorical = [False] * 8


# labels = ['']
target_names = ['Negative', 'Positive']
# labels = ["[{},{})".format(bins[i], bins[i+1]) for i in range(len(bins) - 1)]


def main(name='pima'):

    raw = pd.read_csv(data_path, skiprows=[0], header=None, names=header)
    # target = raw['rings']
    data = raw.drop(columns='Outcome', axis=1).as_matrix()
    target = raw['Outcome'].as_matrix().astype(np.int)

    categories = [None] * len(is_categorical)
    dataset = {
        'target': target,
        'target_names': target_names,
        'is_categorical': is_categorical,
        'is_binary': [False] * len(is_categorical),
        'data': data,
        'feature_names': header[:-1],
        'categories': categories
    }
    save_data(dataset, name)


if __name__ == '__main__':
    main('pima')
