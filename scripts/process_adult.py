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
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""
import numpy as np
from scipy.io import arff

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path


data_name = 'adult'
data_path = get_path('datasets/' + data_name, 'data.arff')

label = 'Risk1Yr'
target_names = ('F', 'T')

meta = {
    'age': ['numeric', None],
    'workclass': ['nominal', (
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked")],
    'fnlwgt': ['nominal', ("divorced", "married", "single", "unknown")],
    'education': ['nominal', (
        "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree",
        "unknown")],
    'default': ['nominal', ("no", "yes", "unknown")],
    'housing': ['nominal', ("no", "yes", "unknown")],
    'loan': ['nominal', ("no", "yes", "unknown")],
    'contact': ['nominal', ("cellular", "telephone")],
    'month': ['nominal', ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")],
    'day_of_week': ['nominal', ("mon", "tue", "wed", "thu", "fri")],
    'duration': ['numeric', None],
    'campaign': ['numeric', None],
    'pdays': ['numeric', None],
    'previous': ['numeric', None],
    'poutcome': ['nominal', "failure","nonexistent","success"],
    'emp.var.rate': ['numeric', None],
    'cons.price.idx': ['numeric', None],
    'cons.conf.idx': ['numeric', None],
    'euribor3m': ['numeric', None],
    'nr.employed': ['numeric', None]
}


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
