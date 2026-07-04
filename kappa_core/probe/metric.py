import torch as t

def binary_classification_metric(pred, answer):
    if pred.ndim == 0:
        return int((pred > 0.5).item()) == answer
    else:
        pred_label = (pred > 0.5).int()
        return pred_label == answer

def multi_classification_metric(pred, answer):
    if pred.ndim == 1:
        pred_class = t.argmax(pred).item()
        return pred_class == answer
    else:
        pred_classes = t.argmax(pred, dim=1)
        return pred_classes == answer 