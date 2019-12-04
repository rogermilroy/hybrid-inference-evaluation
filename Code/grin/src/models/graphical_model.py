from torch import nn
from torch import tensor
import torch

########################################################################################################################
# This code was written with reference to vgsatorras hybrid inference code.
# https://github.com/vgsatorras/hybrid-inference
#
# Almost everything is different but it was invaluable for understanding how to implement the paper:
# Combining Generative and Discriminative Models for Hybrid Inference by Sartorras, Akata and Welling. 20 Jun 2019
#
########################################################################################################################


class KalmanGraphicalModel(nn.Module):
    """
    Class that implements the Kalman Filter as an iterative graph message passing routine.
    Currently just a Smoother. TODO think about what is needed for it to be a filter as well.
    TODO make an implementation of an interface.
    """

    def __init__(self, F: tensor, H: tensor, Q: tensor, R: tensor, standalone: bool = True):
        """
        Initialises the graphical model.

        :param F: The motion model. 2d square tensor
        :param H: The measurement model 2d n x m tensor
        :param Q: The motion noise model 2d square tensor
        :param R: The measurement noise model 2d square tensor
        :param standalone: Whether the model operates on its own or as a component of Hybrid Inference. default True.
        """
        super(KalmanGraphicalModel, self).__init__()
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.negQinv = -Q.inverse()
        self.FtQinv = F.t().matmul(Q.inverse())
        self.HtRinv = H.t().matmul(R.inverse())
        self.standalone = standalone  # TODO is this necessary? Probably not.

    def diff_past_curr(self, x_past: tensor, x_curr: tensor) -> tensor:
        """
        Calculates xt - Fxt-1.
        xcurr and xpast must have the same dimensions.
        should be dims x samples shape.
        :param x_past: xt-1
        :param x_curr: xt
        :return:
        """
        res = x_curr - self.F.matmul(x_past)
        res[0] = 0.  # from first in sequence to itself should always be 0.
        return res

    def diff_curr_fut(self, x_curr: tensor, x_fut: tensor) -> tensor:
        """
        Calculates xt+1 - Fxt
        Should be dims x samples shape.
        :param x_curr:
        :param x_fut:
        :return:
        """
        res = x_fut - self.F.matmul(x_curr)
        res[-1] = 0.
        return res

    def diff_y_curr(self, ys: tensor, x_curr: tensor) -> tensor:
        """
        Calculates y - Hxt
        :param ys:
        :param x_curr:
        :return:
        """
        res = ys - self.H.matmul(x_curr)
        return res

    def once(self, xs, ys):
        """
        Runs a single iteration of the graphical model.
        Returns the messages, xs and ys.
        xs should be state dimensions x number of xs
        ys should be measurement dimensions x number of xs = number of ys
        :param xs: estimates of the state?
        :param ys: observations.
        :return:
        """
        # check dimensions
        if xs.shape[0] == self.F.shape[0]:
            # switch to samples x dim of sample for concatenation.
            xs = xs.t()
        if ys.shape[1] != xs.shape[0]:
            # make sure that the ys are oriented dim x samples
            ys = ys.t()

        # create the time shifted xs. ensure that all are oriented dims x samples now.
        x_past = torch.cat([xs[0].unsqueeze(0), xs[:-1]]).t()
        x_future = torch.cat([xs[1:], xs[-1].unsqueeze(0)]).t()
        xs = xs.t()

        # calculate the differences between the defined paths
        # that is from xt-1 to xt, from xt+1 to xt and from yt to xt.
        # then calculate the messages
        m1 = self.negQinv.matmul(self.diff_past_curr(x_past, xs))
        m2 = self.FtQinv.matmul(self.diff_curr_fut(xs, x_future))
        m3 = self.HtRinv.matmul(self.diff_y_curr(ys, xs))

        # return messages TODO anything else?
        return [m1, m2, m3]

    def forward(self, xs: tensor, ys: tensor, gamma: float, iterations: int = 100):
        """
        This will compute the graphical model solution to the problem.
        Use once() to call a single iteration.
        :param xs: estimates of the states
        :param ys: observations
        :param gamma: the factor by which to update the xs.
        :param iterations: The number of iterations the iterative process should do.
        :return:
        """
        if xs.shape[0] != self.F.shape[0]:
            # switch to samples x dim of sample for concatenation.
            xs = xs.t()
        x = xs
        # iterate up to the number of iterations
        for i in range(iterations):
            # each time calculate the messages
            messages = self.once(x, ys)
            # update the xs by the sum of messages * gamma
            x += sum(messages) * gamma

        # return the result.
        return x