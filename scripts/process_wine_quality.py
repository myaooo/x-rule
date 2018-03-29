"""
Dataset: Wine Quality,
Url: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
Download:
```bash
    mkdir -p ../datasets/wine_quality
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv \
        --output ../datasets/wine_quality/winequality-red.csv
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv \
        --output ../datasets/wine_quality/winequality-white.csv
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names \
        --output ../datasets/wine_quality/winequality.names
```
"""


import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

from sklearn.preprocessing import StandardScaler

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path

base_dir = 'datasets/wine_quality'

# labels = ['']
target_col = 'quality'


def process(color='red'):
    data_path = get_path(base_dir, 'winequality-' + color + '.csv')

    df = pd.read_csv(data_path, sep=';')
    header = list(df.columns)
    # target = raw['rings']
    data_df = df.drop(columns=target_col, axis=1)
    data = data_df.as_matrix()
    target, target_names = process_labels(df, target_col)
    print('target_names', target_names)
    # data[:, 0] = sex
    n_features = data.shape[1]
    categories = [None] * n_features
    is_categorical = [False] * n_features
    dataset = {
        'target': target,
        'target_names': target_names,
        'is_categorical': is_categorical,
        'is_binary': [False] * len(is_categorical),
        'data': data,
        'feature_names': header[:-1],
        'categories': categories
    }
    save_data(dataset, 'wine_quality_' + color)


def process_labels(df, label):
    label_series = df[label].as_matrix()
    max_label = np.max(label_series)
    min_label = np.min(label_series)
    label_series[label_series == min_label] = min_label + 1
    label_series[label_series == max_label] = max_label - 1
    labels = ['level ' + str(l) for l in range(min_label + 1, max_label)]
    return label_series, labels
    # target = np.digitize(label_series, bins) - 1
    # uniq, counts = np.unique(target, return_counts=True)
    # print(counts)
    # return target, labels
#
#
# def process_categorical(df, label):
#     label_series = df[label]
#     label_encoder = LabelEncoder().fit(label_series)
#     return label_encoder.transform(label_series), label_encoder.classes_.tolist()


if __name__ == '__main__':
    process('red')
    process('white')
