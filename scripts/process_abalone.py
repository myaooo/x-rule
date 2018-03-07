"""
Dataset: Abalone,
Url: http://archive.ics.uci.edu/ml/datasets/Abalone
Download:
```bash
    mkdir -p ../datasets/abalone
    curl http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data \
        --output ../datasets/abalone/abalone.data
```


Name / Data Type / Measurement Unit / Description
-----------------------------
Sex / nominal / -- / M, F, and I (infant)
Length / continuous / mm / Longest shell measurement
Diameter	/ continuous / mm / perpendicular to length
Height / continuous / mm / with meat in shell
Whole weight / continuous / grams / whole abalone
Shucked weight / continuous	/ grams / weight of meat
Viscera weight / continuous / grams / gut weight (after bleeding)
Shell weight / continuous / grams / after being dried
Rings / integer / -- / +1.5 gives the age in years

"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

from sklearn.preprocessing import StandardScaler

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path


data_path = get_path('datasets/abalone', 'abalone.data')

header = ['sex', 'length', 'diameter', 'height', 'whole weight',
          'shucked weight', 'viscera weight', 'shell weight', 'rings']

is_categorical = [True, False, False, False, False,
                  False, False, False]


# labels = ['']
bins = [1, 8, 11, 15, 30]
labels = ["[{},{})".format(bins[i], bins[i+1]) for i in range(len(bins) - 1)]


def main():
    raw = pd.read_csv(data_path, header=None, names=header)
    # target = raw['rings']
    data = raw.drop(columns='rings', axis=1).as_matrix()
    target, target_names = process_labels(raw, 'rings')

    sex, sex_names = process_categorical(raw, 'sex')
    data[:, 0] = sex
    categories = [None] * len(is_categorical)
    categories[0] = sex_names
    dataset = {
        'target': target,
        'target_names': target_names,
        'is_categorical': is_categorical,
        'is_binary': [False] * len(is_categorical),
        'data': data,
        'feature_names': header[:-1],
        'categories': categories
    }
    save_data(dataset, 'abalone')


def process_labels(df, label):
    label_series = df[label]
    # label_encoder = LabelEncoder().fit(label_series)
    # return label_encoder.transform(label_series), label_encoder.classes_.tolist()
    target = np.digitize(label_series, bins) - 1
    uniq, counts = np.unique(target, return_counts=True)
    print(counts)
    return target, labels


def process_categorical(df, label):
    label_series = df[label]
    label_encoder = LabelEncoder().fit(label_series)
    return label_encoder.transform(label_series), label_encoder.classes_.tolist()


if __name__ == '__main__':
    main()
