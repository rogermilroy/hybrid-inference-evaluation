from abc import ABC

from torch.nn import Module


class Smoother(Module, ABC):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass
