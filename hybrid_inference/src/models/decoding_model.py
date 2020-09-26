import torch
from torch import nn
from torch.nn import functional
from torch.nn import init

########################################################################################################################
# This code was written with reference to vgsatorras hybrid inference code.
# https://github.com/vgsatorras/hybrid-inference
#
# Implementing the paper:
# Combining Generative and Discriminative Models for Hybrid Inference by Sartorras, Akata and Welling. 20 Jun 2019
#
########################################################################################################################


class GraphDecoder(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(GraphDecoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_in, out_channels=n_hidden, kernel_size=1)
        self.conv1_normed = nn.BatchNorm1d(n_hidden)
        init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.Conv1d(in_channels=n_hidden, out_channels=n_out, kernel_size=1)
        init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        x = functional.leaky_relu(self.conv1_normed(self.conv1(x)))
        x = functional.leaky_relu(self.conv2(x))
        return x
