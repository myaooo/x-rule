"""
Dataset: Mushroom,
Url: http://archive.ics.uci.edu/ml/datasets/Mushroom
Download:
```bash
    mkdir -p ../datasets/mushroom
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names \
        --output ../datasets/mushroom/agaricus-lepiota.names
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data \
        --output ../datasets/mushroom/agaricus-lepiota.data
```

Attribute Information: (classes: edible=e, poisonous=p)
     1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
                                  knobbed=k,sunken=s
     2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
     3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                  pink=p,purple=u,red=e,white=w,yellow=y
     4. bruises?:                 bruises=t,no=f
     5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                  musty=m,none=n,pungent=p,spicy=s
     6. gill-attachment:          attached=a,descending=d,free=f,notched=n
     7. gill-spacing:             close=c,crowded=w,distant=d
     8. gill-size:                broad=b,narrow=n
     9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
                                  green=r,orange=o,pink=p,purple=u,red=e,
                                  white=w,yellow=y
    10. stalk-shape:              enlarging=e,tapering=t
    11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                  rhizomorphs=z,rooted=r,missing=?
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    16. veil-type:                partial=p,universal=u
    17. veil-color:               brown=n,orange=o,white=w,yellow=y
    18. ring-number:              none=n,one=o,two=t
    19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                  none=n,pendant=p,sheathing=s,zone=z
    20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                  orange=o,purple=u,white=w,yellow=y
    21. population:               abundant=a,clustered=c,numerous=n,
                                  scattered=s,several=v,solitary=y
    22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                  urban=u,waste=w,woods=d

"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

from sklearn.preprocessing import StandardScaler

from iml.data_processing import save_data, load_data
from iml.utils.io_utils import get_path

data_path = get_path('datasets/mushroom', 'agaricus-lepiota.data')

header = ['label', 'cap shape', 'cap surface', 'cap color', 'has bruises', 'odor', 'gill attachment', 'gill spacing',
          'gill size', 'gill color', 'stalk shape', 'stalk root', 'stalk surface above ring',
          'stalk surface below ring', 'stalk color above ring', 'stalk color below ring',
          'veil type', 'veil color', 'ring number', 'ring type', 'spore print color', 'population', 'habitat']

raw_feature_map = {
    'label': 'edible=e,poisonous=p',
    'cap shape': 'bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s',
    'cap surface': 'fibrous=f,grooves=g,scaly=y,smooth=s',
    'cap color': 'brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y',
    'has bruises': 'bruises=t,no=f',
    'odor': 'almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s',
    'gill attachment': 'attached=a,descending=d,free=f,notched=n',
    'gill spacing': 'close=c,crowded=w,distant=d',
    'gill size': 'broad=b,narrow=n',
    'gill color': 'black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y',
    'stalk shape': 'enlarging=e,tapering=t',
    'stalk root': 'bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?',
    'stalk surface above ring': 'fibrous=f,scaly=y,silky=k,smooth=s',
    'stalk surface below ring': 'fibrous=f,scaly=y,silky=k,smooth=s',
    'stalk color above ring': 'brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y',
    'stalk color below ring': 'brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y',
    'veil type': 'partial=p,universal=u',
    'veil color': 'brown=n,orange=o,white=w,yellow=y',
    'ring number': 'none=n,one=o,two=t',
    'ring type': 'cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z',
    'spore print color': 'black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y',
    'population': 'abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y',
    'habitat': 'grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d'
}


def parse_features(s: str):
    raws = s.split(',')
    raws = [x.split('=') for x in raws]
    return {cat[1]: cat[0] for cat in raws}


feature_code2name = {label: parse_features(raw_str) for label, raw_str in raw_feature_map.items()}
feature_name2code = {label: {name: code for code, name in code2name.items()} for label, code2name in
                     feature_code2name.items()}

is_categorical = [True] * 22


def main(name='mushroom'):
    target_names = ['edible', 'poisonous']
    raw = pd.read_csv(data_path, header=None, names=header)
    # target = raw['rings']
    data_df = raw.drop(columns='label', axis=1)
    target = process_labels(raw['label'], feature_name2code['label'], target_names)

    categories = [list(feature_name2code[name].keys()) for name in header[1:]]
    data = np.empty(data_df.shape, dtype=np.int)
    for i, feature in enumerate(header[1:]):
        col = data_df[feature]
        d = process_labels(col, feature_name2code[feature], categories[i])
        data[:, i] = d
    is_binary = [True if len(category) == 2 else False for category in categories]
    dataset = {
        'target': target,
        'target_names': target_names,
        'is_categorical': is_categorical,
        'is_binary': is_binary,
        'data': data,
        'feature_names': header[1:],
        'categories': categories
    }
    save_data(dataset, name)


def process_labels(labels, name2code, names):
    target = np.empty((len(labels), ), dtype=np.int)
    for i, name in enumerate(names):
        code = name2code[name]
        target[labels == code] = i
    return target

#
# def process_categorical(df, label):
#     label_series = df[label]
#     label_encoder = LabelEncoder().fit(label_series)
#     return label_encoder.transform(label_series), label_encoder.classes_.tolist()


if __name__ == '__main__':
    main()
