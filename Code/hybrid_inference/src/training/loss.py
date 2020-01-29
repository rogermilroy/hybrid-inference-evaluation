import torch
from torch.nn import functional


def weighted_mse_loss(predictions, targets):
    """
    Custom loss function that weights later iterations more than initial iterations.
    :param predictions: A list of predictions (iterations x sequence_length)
    :param targets: A tensor of target values.
    :return: Tensor a single number tensor of the total loss.
    """
    device = predictions.device
    weights = torch.arange(start=0 + 1/len(predictions), end=1. + 1e-9, step=1/len(predictions), device=device)
    print("Weight len:", len(weights))
    print("Pred len:", len(predictions))
    predictions = predictions.permute(0, 1, 3, 2)
    loss = 0.
    for weight, pred in zip(weights, predictions):
        loss += weight * functional.mse_loss(pred, targets)
    return loss
