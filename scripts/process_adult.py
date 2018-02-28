"""
Dataset: Adult Data Set,
Url: https://archive.ics.uci.edu/ml/datasets/Adult
Download:
```bash
    mkdir -p ../datasets/thoracic
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -o ../datasets/adult/adult.data
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -o ../datasets/adult/adult.test
```
Attributes


age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters,
    1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners,
    Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India,
    Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal,
    Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
    Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""

from functools import reduce
import numpy as np
import pandas as pd

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path, save_file


data_name = 'adult'
train_data_path = get_path('datasets/' + data_name, 'adult.data')
test_data_path = get_path('datasets/' + data_name, 'adult.test')

target_names = ('<=50K', '>50K')

meta = {
    'age': ['numeric', None],
    'workclass': ['nominal', (
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked")],
    'fnlwgt': ['numeric', None],
    'education': ['nominal', (
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th",
        "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")],
    'education-num': ['numeric', None],
    'marital-status': ['nominal', ("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                                   "Married-spouse-absent", "Married-AF-spouse")],
    'occupation': ['nominal', ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                               "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                               "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                               "Armed-Forces")],
    'relationship': ['nominal', ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")],
    'race': ['nominal', ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")],
    'sex': ['nominal', ("Female", "Male")],
    'capital-gain': ['numeric', None],
    'capital-loss': ['numeric', None],
    'hours-per-week': ['numeric', None],
    'native-country': ['nominal', ("United-States", "Others")],
    'target': ['nominal', target_names]
}

header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
          'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']


def main():

    train = pd.read_csv(train_data_path, header=None, names=header, skipinitialspace=True, na_values="?")\
        .drop(columns='education', axis=1).dropna()
    test = pd.read_csv(test_data_path, header=None, skiprows=[0], names=header, skipinitialspace=True, na_values="?")\
        .drop(columns='education', axis=1).dropna()
    test['target'] = [s[:-1] for s in test['target']]
    attrs = list(train.columns)

    print(len(train))
    print(len(test))
    is_categorical = [True if meta[attr][0] == 'nominal' else False for attr in attrs[:-1]]
    is_binary = [True if (meta[attr][0] == 'nominal' and len(meta[attr][1]) == 2) else False for attr in attrs[:-1]]

    def process_data(df):
        countries = df['native-country'].copy()
        countries[countries != 'United-States'] = 'Others'
        df['native-country'] = countries
        _data = np.full((len(df), len(attrs)), np.nan)
        for i, attr in enumerate(attrs):
            col = df[attr]
            if i == len(attrs) - 1 or is_categorical[i]:
                col = process_categorical(np.array(col), meta[attr][1])
            _data[:, i] = col
        x = _data[:, :-1]
        y = _data[:, -1].astype(np.int)
        return x, y

    train_x, train_y = process_data(train)
    test_x, test_y = process_data(test)
    data = np.vstack((train_x, test_x))
    target = np.hstack((train_y, test_y))

    print(len(train_x))
    print(len(test_x))
    categories = [meta[attr][1] if is_cat else None for attr, is_cat in zip(attrs[:-1], is_categorical)]
    dataset = {
        'target': target,
        'target_names': target_names,
        'is_categorical': is_categorical,
        'is_binary': is_binary,
        'data': data,
        'feature_names': attrs[:-1],
        'categories': categories
    }
    save_data(dataset, data_name)
    save_file(train_x, get_path('datasets/' + data_name, 'train_x.npy'))
    save_file(train_y, get_path('datasets/' + data_name, 'train_y.npy'))
    save_file(test_x, get_path('datasets/' + data_name, 'test_x.npy'))
    save_file(test_y, get_path('datasets/' + data_name, 'test_y.npy'))

# def process_target(col, labels):
#     label_series = df[label]
#     label_encoder = LabelEncoder().fit(label_series)
#     # return label_encoder.transform(label_series), label_encoder.classes_.tolist()
#     target = np.digitize(label_series, bins) - 1
#     return target, labels


def process_categorical(col, categories):
    data = np.full(col.shape, np.nan)
    for i, cat in enumerate(categories):
        logic = col == cat
        data[logic] = i
    return data


if __name__ == '__main__':
    main()
