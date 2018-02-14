import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

from sklearn.preprocessing import StandardScaler

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path

data_path = get_path('datasets/diabetes', 'diabetic_data.csv')

label_column = 'readmitted'

columns_to_drop = ['weight', 'encounter_id', 'patient_nbr', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3'] + [label_column]

numeric_features = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses']


def process_labels(df, label):
    label_series = df[label]
    label_encoder = LabelEncoder().fit(label_series)
    return label_encoder.transform(label_series), label_encoder.classes_.tolist()


def process_data(df, droped_cols, numeric_features):
    data_df = df.drop(columns=droped_cols)

    num_data_df = data_df[numeric_features]
    num_data = num_data_df.as_matrix()

    cat_data_df = data_df.drop(columns=numeric_features)
    categorical_features = list(cat_data_df.columns)

    encoders = []
    cat_data_list = []
    for categorical_feature in categorical_features:
        data_col = data_df[categorical_feature]
        encoder = LabelEncoder().fit(data_col)
        encoders.append(encoder)
        cat_data_list.append(encoder.transform(data_col))
        print("Feature:", categorical_feature, ", #categories:", len(encoder.classes_))

    cat_data = np.array(cat_data_list).T
    print(cat_data.shape, num_data.shape)

    data = np.hstack([num_data, cat_data.astype(np.float)])
    feature_names = list(num_data_df.columns) + categorical_features
    descriptions = {feature_name: encoder.classes_.tolist() for feature_name, encoder in zip(categorical_features, encoders)}
    is_categorical = [0] * len(numeric_features) + [1] * len(categorical_features)
    is_categorical = np.array(is_categorical, dtype=bool)
    return data, feature_names, is_categorical, descriptions


def standardize(data, features):
    xScaler = StandardScaler().fit(data[features].toarray())
    scaled = xScaler.transform(data[features].toarray())
    new_data = data.copy()
    new_data[features] = scaled
    return new_data, xScaler


# def encode_one_hot(data, categorical_features):
#     encoder = OneHotEncoder(categorical_features=categorical_features).fit(data)
#
#     return encoder.transform(data), encoder


def main():
    df = pd.read_csv(data_path)
    print("Feature names:", list(df.columns))

    target, target_names = process_labels(df, label_column)
    data, feature_names, is_categorical, descriptions = process_data(df, columns_to_drop, numeric_features)

    dataset = {
        'data': data,
        'target': target,
        'target_names': target_names,
        'feature_names': feature_names,
        'is_categorical': is_categorical,
        'descriptions': descriptions,
    }
    save_data(dataset, 'diabetes')
    dataset = load_data('diabetes')
    for key, val in dataset.items():
        print(key)
        # print(val)
        print(type(val))
        if isinstance(val, np.ndarray):
            print(val.dtype, val.shape)
        print('')


if __name__ == '__main__':
    main()

