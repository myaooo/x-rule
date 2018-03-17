"""
Dataset: Thoracic Surgery Data,
Url: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
Download:
```bash
    mkdir -p ../datasets/bank_marketing
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip \
        -o ../datasets/bank_marketing/bank-additional.zip
    unzip ../datasets/bank_marketing/bank-additional.zip -d ../datasets/bank_marketing

    sed -i'.back' 's/;/,/g' ../datasets/bank_marketing/bank-additional/bank-additional.csv
    sed -i'.back' 's/;/,/g' ../datasets/bank_marketing/bank-additional/bank-additional-full.csv
    
```

   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
   3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
   5 - default: has credit in default? (categorical: "no","yes","unknown")
   6 - housing: has housing loan? (categorical: "no","yes","unknown")
   7 - loan: has personal loan? (categorical: "no","yes","unknown")
   # related with the last contact of the current campaign:
   8 - contact: contact communication type (categorical: "cellular","telephone")
   9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
  11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
   # other attributes:
  12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
  14 - previous: number of contacts performed before this campaign and for this client (numeric)
  15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
   # social and economic context attributes
  16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
  17 - cons.price.idx: consumer price index - monthly indicator (numeric)
  18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
  19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
  20 - nr.employed: number of employees - quarterly indicator (numeric)

  21 - y - has the client subscribed a term deposit? (binary: "yes","no")

"""

import numpy as np
import pandas as pd
from scipy.io import arff

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path

data_name = 'bank_marketing'
data_path = get_path('datasets/' + data_name + '/bank-additional', 'bank-additional.csv')

label = 'y'
target_names = ('no', 'yes')

meta = {
    'age': ['numeric', None],
    'job': ['nominal', (
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown")],
    'marital': ['nominal', ("divorced", "married", "single", "unknown")],
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
    df = pd.read_csv(data_path)
    attrs = list(df.columns)[:-1]
    is_binary = [True if (meta[attr][1] is not None and len(meta[attr][1]) == 2) else False for attr in attrs]
    is_categorical = [True if meta[attr][0] == 'nominal' else False for attr in attrs]

    data = []
    for i, attr in enumerate(attrs):
        col = df[attr]
        if is_categorical[i]:
            col = process_categorical(np.array(col), meta[attr][1])
        data.append(col)
    data = np.vstack(data)
    target = process_categorical(df[label], target_names)
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
        logic = col == cat
        data[logic] = i
    return data


if __name__ == '__main__':
    main()
