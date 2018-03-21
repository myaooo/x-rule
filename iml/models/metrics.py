from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score


def label2binary(y):
    return OneHotEncoder().fit_transform(y.reshape([-1, 1])).toarray()


def auc_score(y_true, y_pred, average=None):
    return roc_auc_score(label2binary(y_true), y_pred, average=average)


