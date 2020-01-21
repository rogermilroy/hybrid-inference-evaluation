import torch
from torch.nn import functional


def weighted_mse_loss(predictions, targets):
    """
    Custom loss function that weights later iterations more than initial iterations.
    :param predictions: A list of predictions (iterations x sequence_length)
    :param targets: A tensor of target values.
    :return: Tensor a single number tensor of the total loss.
    """
    weights = torch.arange(start=0, end=1. + 1e-7, step=1/len(predictions))
    print("Weight len:", len(weights))
    print("Pred len:", len(predictions))
    loss = 0.
    for weight, pred in zip(predictions, weights):
        loss += weight * functional.mse_loss(pred, targets)
    return loss
