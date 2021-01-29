import tensorflow.keras.backend as K


def recall(y_true, y_pred):
    """
    Function for computing the recall score using tensors. The Function
    handles only the binary case and compute recall for the positive class
    only.

    Args:
        - y_true: keras tensor, ground truth labels
        - y_pred: keras tensord, labels estimated by the model

    Returns:
        - recall: float, recall score for the positive class
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """
    Function for computing the precision score using tensors. The Function
    handles only the binary case and compute precision for the positive class
    only.

    Args:
        - y_true: keras tensor, ground truth labels
        - y_pred: keras tensord, labels estimated by the model

    Returns:
        - precision: float, precision score for the positive class
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    """
    Function for computing the unweighted f1 score using tensors. The Function
    handles only the binary case and compute the unweighted f1 score
    for the positive class only.

    Args:
        - y_true: keras tensor, ground truth labels
        - y_pred: keras tensord, labels estimated by the model

    Returns:
        - f1: float, unweighted f1 score for the positive class
    """
    precision_v = precision(y_true, y_pred)
    recall_v = recall(y_true, y_pred)
    nominator = 2 * (precision_v * recall_v)
    denominator = (precision_v + recall_v + K.epsilon())
    f1 = nominator / denominator
    return f1
