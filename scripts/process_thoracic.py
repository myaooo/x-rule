"""
Dataset: Thoracic Surgery Data,
Url: https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data
Download:
```bash
    mkdir -p ../datasets/thoracic
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/00277/ThoraricSurgery.arff \
        --output ../datasets/thoracic/data.arff
```
"""
import numpy as np
from scipy.io import arff

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path


data_name = 'thoracic'
data_path = get_path('datasets/' + data_name, 'data.arff')

label = 'Risk1Yr'
target_names = ('F', 'T')


def main():
    raw, meta = arff.loadarff(data_path)
    attrs = meta.names()[:-1]
    is_binary = [True if (meta[attr][1] is not None and len(meta[attr][1]) == 2) else False for attr in attrs]
    is_categorical = [True if attr_type == 'nominal' else False for attr_type in meta.types()[:-1]]

    data = []
    for i, attr in enumerate(attrs):
        col = raw[attr]
        if is_categorical[i]:
            col = process_categorical(np.array(col), meta[attr][1])
        data.append(col)
    data = np.vstack(data)
    target = process_categorical(raw[label], target_names)
    # is_categorical = np.logical_and(is_categorical, np.logical_not(is_binary))

    categories = [meta[attr][1] if is_cat else None for attr, is_cat in zip(attrs, is_categorical)]
    dataset = {
        'target': target,
        'target_names': target_names,
        'is_categorical': is_categorical,
        'is_binary': is_binary,
        'data': data.T,
        'feature_names': attrs,
        'categories': categories
    }
    save_data(dataset, data_name)


# def process_target(col, labels):
#     label_series = df[label]
#     label_encoder = LabelEncoder().fit(label_series)
#     # return label_encoder.transform(label_series), label_encoder.classes_.tolist()
#     target = np.digitize(label_series, bins) - 1
#     return target, labels


def process_categorical(col, categories):
    data = np.zeros(col.shape, dtype=np.int32)
    for i, cat in enumerate(categories):
        logic = col == bytes(cat.encode('ascii'))
        data[logic] = i
    return data


if __name__ == '__main__':
    main()
