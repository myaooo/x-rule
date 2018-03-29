"""
Dataset: Breast Cancer Original,
Url: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
Download:
```bash
    mkdir -p ../datasets/breast_cancer_original
    curl http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data \
        --output ../datasets/breast_cancer_original/breast-cancer-wisconsin.data
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names \
        --output ../datasets/breast_cancer_original/breast-cancer-wisconsin.names
```

Attribute Information: (classes: edible=e, poisonous=p)
    1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)

"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

from sklearn.preprocessing import StandardScaler

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path

data_path = get_path('datasets/breast_cancer_original', 'breast-cancer-wisconsin.data')

header = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
          'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
          'Normal Nucleoli', 'Mitoses', 'label']


is_categorical = [False] * 9


def main(name='breast_cancer_original'):
    target_names = ['edible', 'poisonous']
    raw = pd.read_csv(data_path, header=None, names=header, na_values=['?']).fillna(0)
    # target = raw['rings']
    raw = raw.drop(columns='ID')
    data_df = raw.drop(columns='label', axis=1)
    target = process_labels(raw['label'], [2, 4])
    target_names = ['benign', 'malignant']

    categories = [None] * 9
    data = data_df.as_matrix()
    is_binary = [False] * 9
    dataset = {
        'target': target,
        'target_names': target_names,
        'is_categorical': is_categorical,
        'is_binary': is_binary,
        'data': data,
        'feature_names': header[2:],
        'categories': categories
    }
    save_data(dataset, name)


def process_labels(labels, codes):
    target = np.empty((len(labels), ), dtype=np.int)
    for i, code in enumerate(codes):
        target[labels == code] = i
    return target

#
# def process_categorical(df, label):
#     label_series = df[label]
#     label_encoder = LabelEncoder().fit(label_series)
#     return label_encoder.transform(label_series), label_encoder.classes_.tolist()


if __name__ == '__main__':
    main()
