import numpy as np
import torch


def np2torch(arr: np.ndarray) -> torch.tensor:
    """
    Conversion utility that takes an numpy ndarray and returns a PyTorch tensor.
    Be careful with data types!!!! Always fully define the data type eg. float64 not just float. Same with int etc.
    :param arr: ndarray. The input numpy array.
    :return: tensor. The tensor representation of the array.
    """
    return torch.from_numpy(arr)


def torch2np(tens: torch.tensor) -> np.ndarray:
    """
    Conversion utility that takes a Pytorch tensor and returns a numpy ndarray.
    Be careful with datatypes!!! Always fully define the data type eg. float64 not just float. Same with int etc.
    :param tens: tensor. The input tensor
    :return: ndarray. The numpy ndarray representation of the tensor. Or Exception if you don't detach.
    """
    return tens.cpu().detach().numpy()
