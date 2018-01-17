import numpy as np


def split(data, ratio, shuffle=False):

    try:
        ratio = np.cumsum(ratio)
        ratio /= ratio[-1]
        split_indices = [int(i * len(data)) for i in ratio[:-1]]
    except TypeError as e:
        print("ratio must be an iterable")
        raise e

    if shuffle:
        np.random.shuffle(data)
    return np.split(data, split_indices)

