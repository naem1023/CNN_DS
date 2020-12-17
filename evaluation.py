from keras.metrics import Precision, Recall
from keras import backend as K

def get_precision(y_true, y_predict):
    m = Precision()
    m.update_state(y_true, y_predict)

    return m.result().numpy()

def get_recall(y_true, y_predict):
    m = Recall()
    m.update_state(y_true, y_predict)

    return m.result().numpy()

def get_f1score(precision, recall):
    # K.epsilon() for preventing 'divide by zero error'
    f1score = (2 * recall * precision) / (recall + precision + K.epsilon())

    return f1score

def evaluate(y_true, y_predict):
    precision = get_precision(y_true, y_predict)
    recall = get_recall(y_true, y_predict)
    f1score = get_f1score(precision, recall)

    return precision, recall, f1score


