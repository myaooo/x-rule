"""
Dataset: default of credit card clients Data Set,
Url: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
Download:
```bash
    mkdir -p ../datasets/credit_card
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls \
        -o ../datasets/credit_card/data.xls
```
"""
import numpy as np
import pandas as pd

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path


data_name = 'credit_card'
data_path = get_path('datasets/' + data_name, 'data.xls')

# label = 'Risk1Yr'

feature_map = {
    'X1': ('Limit Balance', 'numeric', None), 
    'X2': ('Gender', 'nominal', ('male', 'female')), 
    'X3': ('Education', 'nominal', ('graduate school', 'university', 'high school', 'others')), 
    'X4': ('Marriage', 'nominal', ('married', 'single', 'others')),
    'X5': ('AGE', 'numeric', None), 
    'X6': ('Pay Delay Sep', 'numeric', None),
    'X7': ('Pay Delay Aug', 'numeric', None), 
    'X8': ('Pay Delay Jul', 'numeric', None), 
    'X9': ('Pay Delay Jun', 'numeric', None), 
    'X10': ('Pay Delay May', 'numeric', None), 
    'X11': ('Pay Delay Apr', 'numeric', None), 
    'X12': ('Bill Amount Sep', 'numeric', None), 
    'X13': ('Bill Amount Aug', 'numeric', None), 
    'X14': ('Bill Amount Jul', 'numeric', None), 
    'X15': ('Bill Amount Jun', 'numeric', None), 
    'X16': ('Bill Amount May', 'numeric', None), 
    'X17': ('Bill Amount Apr', 'numeric', None), 
    'X18': ('Pay Amount Sep', 'numeric', None), 
    'X19': ('Pay Amount Aug', 'numeric', None), 
    'X20': ('Pay Amount Jul', 'numeric', None), 
    'X21': ('Pay Amount Jun', 'numeric', None), 
    'X22': ('Pay Amount May', 'numeric', None), 
    'X23': ('Pay Amount Apr', 'numeric', None), 
    'Y': ('Default Payment', 'nominal', ('Payment', 'Default')),
}

feature_names = [feature_map['X' + str(i)][0] for i in range(1, 24)]
categories = [feature_map['X' + str(i)][2] for i in range(1, 24)]
is_binary = [False] * 23
is_binary[1] = True
is_categorical = [feature_map['X' + str(i)][1] == 'nominal' for i in range(1, 24)]
target_names = feature_map['Y'][2]


def main():
    df = pd.read_excel(data_path, skiprows=[1], index_col=0)
    # Deal with strange value here
    df['X3'][df['X3'] > 3] = 4
    df['X3'][df['X3'] == 0] = 4
    df['X4'][df['X4'] == 0] = 3
    mat = df.as_matrix()
    mat[:, 1:4] -= 1
    data = mat[:, :-1]
    target = mat[:, -1]

    dataset = {
        'target': target,
        'target_names': target_names,
        'is_categorical': is_categorical,
        'is_binary': is_binary,
        'data': data,
        'feature_names': feature_names,
        'categories': categories
    }
    save_data(dataset, data_name)


if __name__ == '__main__':
    main()
