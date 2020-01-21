from torch import nn
from torch import tensor
from .graphical_model import KalmanGraphicalModel
from .graphical_nn_model import KalmanGNN

########################################################################################################################
# This code was written with reference to vgsatorras hybrid inference code.
# https://github.com/vgsatorras/hybrid-inference
#
# Almost everything is different but it was invaluable for understanding how to implement the paper:
# Combining Generative and Discriminative Models for Hybrid Inference by Sartorras, Akata and Welling. 20 Jun 2019
#
########################################################################################################################


class HybridInference(nn.Module):

    def __init__(self, F: tensor, H: tensor, Q: tensor, R: tensor, gamma: float):
        super(HybridInference, self).__init__()
        self.graph = KalmanGraphicalModel(F=F, H=H, Q=Q, R=R)
        self.gnn = KalmanGNN(h_dim=48, x_dim=F.shape[0], y_dim=R.shape[0])
        self.H = H
        self.gamma = gamma

    def forward(self, ys, iterations=50):
        """
        Forward pass through the hybrid inferrer.
        This is an iterative procedure that progressively refines the estimate of the xs (states)
        First the xs are estimated by running the ys (observations) backwards through the measurement matrix.
        (Aka multiply by the transpose).
        After that initialise hx (hidden states of the gnn nodes) and hy.
        hx is random and ys are convolved to get hys that are the same dimension as hx.
        In each iteration the graphical model messages are computed by calling graph.once().
        See KalmanGraphicalModel for more info.
        These messages are passed to the gnn along with hx. The new hx and a correction eps are returned.
        The estimates of state xs are then updated with the sum of the graphical model messages and eps weighted by gamma.
        :param ys:
        :param iterations:
        :return:
        """
        xs = self.H.t().matmul(ys.t())

        hx = self.gnn.initialise_hx_y(ys)

        for i in range(iterations):
            # compute the graphical model messages
            messages = self.graph.once(xs, ys)
            # compute the hidden states and epsilon correction
            eps, hx = self.gnn(hx, messages)
            # update the xs with messages and epsilon.
            xs = xs + self.gamma * (eps + sum(messages))
        # return the final estimate of the positions.
        return xs
