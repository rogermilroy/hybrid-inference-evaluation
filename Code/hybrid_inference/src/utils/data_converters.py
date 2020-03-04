import torch
import numpy as np


def torch2numpy(tens: torch.tensor) -> np.ndarray:
    return tens.numpy()


def numpy2torch(arr: np.ndarray) -> torch.tensor:
    return torch.from_numpy(arr).to(torch.float32)  # needs to be float 32 to match targets


def torchseq2numpyseq(seq: torch.tensor):
    seq = seq.squeeze()
    l = list()
    for tens in seq:
        l.append(torch2numpy(tens))
    return l
