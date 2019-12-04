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


class GraphEncoderMLP(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(GraphEncoderMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=n_in, out_features=n_hidden)
        init.xavier_normal_(self.fc1.weight)
        self.fc1_normed = nn.BatchNorm1d(num_features=n_hidden)
        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_out)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = functional.leaky_relu(self.fc1_normed(self.fc1(x)))
        x = functional.leaky_relu(self.fc2(x))
        return x
