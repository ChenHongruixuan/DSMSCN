import keras.backend as K


def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (possible_negatives + K.epsilon())


def F1_score(y_true, y_pred):
    R = Recall(y_true, y_pred)
    P = Precision(y_true, y_pred)
    return 2 * P * R / (R + P)
