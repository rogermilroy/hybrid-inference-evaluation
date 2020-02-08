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
    Currently just a Smoother.
    TODO make an implementation of an interface.
    """

    def __init__(self, F: tensor, H: tensor, Q: tensor, R: tensor):
        """
        Initialises the graphical model.

        :param F: The motion model. 2d square tensor
        :param H: The measurement model 2d n x m tensor
        :param Q: The motion noise model 2d square tensor
        :param R: The measurement noise model 2d square tensor
        """
        super(KalmanGraphicalModel, self).__init__()
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.negQinv = -Q.inverse()
        self.FtQinv = F.t().matmul(Q.inverse())
        self.HtRinv = H.t().matmul(R.inverse())

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

    def iterate(self, xs: tensor, ys: tensor, gamma: float, iterations: int = 100):
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
            messages = self.forward(x, ys)
            # update the xs by the sum of messages * gamma
            x += sum(messages) * gamma

        # return the result.
        return x

    def forward(self, xs: tensor, ys: tensor):
        """
        Runs a single iteration of the graphical model.
        Returns the messages, xs and ys.
        xs should be batch x feat x samples
        ys should be batch x feat x samples
        :param xs: estimates of the state
        :param ys: observations.
        :return:
        """

        # TODO check each dim is what it should be. For now just do it right.
        # change to samples x batch x feat for manipulation.
        xs = xs.permute(2, 0, 1)

        # create the time shifted xs. ensure that all are oriented batch x feat x samples now.
        x_past = torch.cat([xs[0].unsqueeze(0), xs[:-1]]).permute(1, 2, 0)
        x_future = torch.cat([xs[1:], xs[-1].unsqueeze(0)]).permute(1, 2, 0)
        xs = xs.permute(1, 2, 0)

        # make sure that the ys are oriented batch x feat x samples
        ys = ys.permute(0, 2, 1)

        # result dims batch x feat x samples
        m1 = self.negQinv.matmul(self.diff_past_curr(x_past, xs))
        m2 = self.FtQinv.matmul(self.diff_curr_fut(xs, x_future))
        m3 = self.HtRinv.matmul(self.diff_y_curr(ys, xs))


        # return messages
        return [m1, m2, m3]


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


class KalmanInputGraphicalModel(nn.Module):
    """
    Class that implements the Kalman Filter as an iterative graph message passing routine.
    Currently just a Smoother.
    TODO make an implementation of an interface.
    """

    def __init__(self, F: tensor, H: tensor, Q: tensor, R: tensor, G: tensor):
        """
        Initialises the graphical model.

        :param F: The motion model. 2d square tensor
        :param H: The measurement model 2d n x m tensor
        :param Q: The motion noise model 2d square tensor
        :param R: The measurement noise model 2d square tensor
        """
        super(KalmanInputGraphicalModel, self).__init__()
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.G = G
        self.negQinv = -Q.inverse()
        self.FtQinv = F.t().matmul(Q.inverse())
        self.HtRinv = H.t().matmul(R.inverse())

    def diff_past_curr(self, x_past: tensor, x_curr: tensor, us: tensor) -> tensor:
        """
        Calculates xt - Fxt-1.
        xcurr and xpast must have the same dimensions.
        should be dims x samples shape.
        :param x_past: xt-1
        :param x_curr: xt
        :param us: ut
        :return:
        """
        res = x_curr - self.F.matmul(x_past) + self.G.matmul(us)
        return res

    def diff_curr_fut(self, x_curr: tensor, x_fut: tensor, us_fut: tensor) -> tensor:
        """
        Calculates xt+1 - Fxt
        Should be dims x samples shape.
        :param x_curr: xt
        :param x_fut: xt+1
        :param us_fut: ut+1
        :return:
        """
        res = x_fut - (self.F.matmul(x_curr) + self.G.matmul(us_fut))
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

    def iterate(self, xs: tensor, ys: tensor, us: tensor, gamma: float, iterations: int = 100):
        """
        This will compute the graphical model solution to the problem.
        Use once() to call a single iteration.
        :param xs: estimates of the states
        :param ys: observations
        :param us: the inputs.
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
            messages = self.forward(x, ys, us)
            # update the xs by the sum of messages * gamma
            x += sum(messages) * gamma

        # return the result.
        return x

    def forward(self, xs: tensor, ys: tensor, us: tensor):
        """
        Runs a single iteration of the graphical model.
        Returns the messages, xs and ys.
        xs should be batch x feat x samples
        ys should be batch x feat x samples
        :param xs: estimates of the state
        :param ys: observations.
        :param us: inputs
        :return:
        """

        # TODO check each dim is what it should be. For now just do it right.
        # start as batch x feat x samples ( this is set in HI would be nice to change.)
        # change to samples x batch x feat for manipulation.
        xs = xs.permute(2, 0, 1)

        # create the time shifted xs. ensure that all are oriented batch x feat x samples now.
        x_past = torch.cat([xs[0].unsqueeze(0), xs[:-1]]).permute(1, 2, 0)
        x_future = torch.cat([xs[1:], xs[-1].unsqueeze(0)]).permute(1, 2, 0)
        xs = xs.permute(1, 2, 0)

        # batch x samples x feat
        us = us.permute(1, 0, 2)

        # create the time shifted xs. ensure that all are oriented batch x feat x samples now.
        us_fut = torch.cat([us[1:], us[-1].unsqueeze(0)]).permute(1, 2, 0)
        us = us.permute(1, 2, 0)

        # make sure that the ys are oriented batch x feat x samples
        ys = ys.permute(0, 2, 1)

        # result dims batch x feat x samples
        m1 = self.negQinv.matmul(self.diff_past_curr(x_past, xs, us))
        m2 = self.FtQinv.matmul(self.diff_curr_fut(xs, x_future, us_fut))
        m3 = self.HtRinv.matmul(self.diff_y_curr(ys, xs))

        # return messages
        return [m1, m2, m3]

