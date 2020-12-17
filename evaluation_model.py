from keras.metrics import Precision, Recall
from keras import backend as K

from sklearn.metrics import precision_score, recall_score, f1_score

def get_precision(y_true, y_predict):
    return precision_score(y_true, y_predict, average='macro')

def get_recall(y_true, y_predict):
    return recall_score(y_true, y_predict, average='macro')

def get_f1score(y_true, y_predict):
    return f1_score(y_true, y_predict, average='macro')

def evaluate(y_true, y_predict):
    precision = get_precision(y_true, y_predict)
    recall = get_recall(y_true, y_predict)
    f1score = get_f1score(y_true, y_predict)

    return precision, recall, f1score


