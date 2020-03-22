import torch
from src.models.predictor import Predictor
from src.models.smoother import Smoother


####################################################################################################
# This code was written with reference to vgsatorras hybrid inference code.
# https://github.com/vgsatorras/hybrid-inference
#
# Almost everything is different but it was invaluable for understanding how to implement the paper:
# Combining Generative and Discriminative Models for Hybrid Inference
# by Satorras, Akata and Welling. 20 Jun 2019
#
####################################################################################################


class KalmanGraphicalModel(Smoother, Predictor):
    """
    Class that implements the Kalman Filter as an iterative graph message passing routine.
    Currently just a Smoother.
    TODO make an implementation of an interface.
    """

    def __init__(self, F, H, Q, R):
        """
        Initialises the graphical model.

        :param F: The motion model. 2d square tensor
        :param H: The measurement model 2d n x m tensor
        :param Q: The motion noise model 2d square tensor
        :param R: The measurement noise model 2d square tensor
        """
        super().__init__()
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.negQinv = -Q.inverse()
        self.FtQinv = F.t().matmul(Q.inverse())
        self.HtRinv = H.t().matmul(R.inverse())

    def diff_past_curr(self, x_past, x_curr):
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

    def diff_curr_fut(self, x_curr, x_fut):
        """
        Calculates xt+1 - Fxt
        Should be dims x samples shape.
        :param x_curr:
        :param x_fut:
        :return:
        """
        res = x_fut - self.F.matmul(x_curr)
        return res

    def diff_y_curr(self, ys, x_curr):
        """
        Calculates y - Hxt
        :param ys:
        :param x_curr:
        :return:
        """
        res = ys - self.H.matmul(x_curr)
        return res

    def iterate(self, xs, ys, gamma: float, iterations: int = 200):
        """
        This will compute the graphical model solution to the problem.
        Use once() to call a single iteration.
        :param xs: estimates of the states
        :param ys: observations
        :param gamma: the factor by which to update the xs.
        :param iterations: The number of iterations the iterative process should do.
        :return:
        """
        # xs should be batch x feat x samples
        # ys should be batch x feat x samples.
        x = xs
        # iterate up to the number of iterations
        for i in range(iterations):
            # each time calculate the messages
            messages = self.forward(x, ys)
            # update the xs by the sum of messages * gamma
            x += sum(messages) * gamma

        # return the result.
        return x

    def forward(self, xs, ys):
        """
        Runs a single iteration of the graphical model.
        Returns the messages, xs and ys.
        xs should be (batch x feat x samples)
        ys should be (batch x feat x samples)
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
        if ys.shape[2] != xs.shape[2]:
            ys = ys.permute(0, 2, 1)

        # result dims batch x feat x samples
        m1 = self.negQinv.matmul(self.diff_past_curr(x_past, xs))
        m2 = self.FtQinv.matmul(self.diff_curr_fut(xs, x_future))
        m3 = self.HtRinv.matmul(self.diff_y_curr(ys, xs))

        # return messages
        return [m1, m2, m3]

    def predict(self, n: int, xs, ys, gamma: float, iterations: int = 200):
        """
        Predicts up to n time steps into the future.
        inputs expected to be (batch x feat x seq)
        :param n:
        :param xs:
        :param ys:
        :param gamma:
        :param iterations:
        :return:
        """
        # add n steps to the end of ys. (should be sequence dimension)
        ys.pad((0, n), 'constant', 0)

        return self.iterate(xs=xs, ys=ys, gamma=gamma, iterations=iterations)


####################################################################################################
# This code was written with reference to vgsatorras hybrid inference code.
# https://github.com/vgsatorras/hybrid-inference
#
# Almost everything is different but it was invaluable for understanding how to implement the paper:
# Combining Generative and Discriminative Models for Hybrid Inference
# by Satorras, Akata and Welling. 20 Jun 2019
#
####################################################################################################


class KalmanInputGraphicalModel(Smoother, Predictor):
    """
    Class that implements the Kalman Filter as an iterative graph message passing routine.
    Currently just a Smoother.
    TODO make an implementation of an interface.
    """

    def __init__(self, F, H, Q, R, G):
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

    def diff_past_curr(self, x_past, x_curr, us):
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

    def diff_curr_fut(self, x_curr, x_fut, us_fut):
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

    def diff_y_curr(self, ys, x_curr):
        """
        Calculates y - Hxt
        :param ys:
        :param x_curr:
        :return:
        """
        res = ys - self.H.matmul(x_curr)
        return res

    def iterate(self, xs, ys, us, gamma: float, iterations: int = 200):
        """
        This will compute the graphical model solution to the problem.
        Use once() to call a single iteration.
        (batch x feat x seq)
        :param xs: estimates of the states
        :param ys: observations
        :param us: the inputs.
        :param gamma: the factor by which to update the xs.
        :param iterations: The number of iterations the iterative process should do.
        :return:
        """
        x = xs
        # iterate up to the number of iterations
        for i in range(iterations):
            # each time calculate the messages
            messages = self.forward(x, ys, us)
            # update the xs by the sum of messages * gamma
            x += sum(messages) * gamma

        # return the result.
        return x

    def forward(self, xs, ys, us):
        """
        Runs a single iteration of the graphical model.
        Returns the messages, xs and ys.
        xs should be (batch x feat x samples)
        ys should be (batch x feat x samples)
        us should be (batch x feat x samples)
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

        #  samples x batch x feat
        us = us.permute(2, 0, 1)

        # create the time shifted xs. ensure that all are oriented batch x feat x samples now.
        us_fut = torch.cat([us[1:], us[-1].unsqueeze(0)]).permute(1, 2, 0)
        us = us.permute(1, 2, 0)

        # make sure that the ys are oriented batch x feat x samples
        if ys.shape[2] != xs.shape[2]:
            ys = ys.permute(0, 2, 1)

        # result dims batch x feat x samples
        m1 = self.negQinv.matmul(self.diff_past_curr(x_past, xs, us))
        m2 = self.FtQinv.matmul(self.diff_curr_fut(xs, x_future, us_fut))
        m3 = self.HtRinv.matmul(self.diff_y_curr(ys, xs))

        # return messages
        return [m1, m2, m3]

    def predict(self, n: int, xs, ys, us, gamma: float,
                iterations: int = 200):
        """
        Predicts up to n time steps into the future.
        inputs expected to be (batch x feat x seq)
        :param n:
        :param xs:
        :param ys:
        :param us:
        :param gamma:
        :param iterations:
        :return:
        """
        # us must match size otherwise error
        if us.shape[2] != (ys.shape[2] + n):
            raise Exception("Inputs (us) must be provided to line up with predicted time.")
        else:
            # add n steps to the end of ys. (should be sequence dimension)
            ys.pad((0, n), 'constant', 0)

        return self.iterate(xs=xs, ys=ys, us=us, gamma=gamma, iterations=iterations)


####################################################################################################
#  EKF version (takes F as input for each forward pass.
####################################################################################################


class ExtendedKalmanGraphicalModel(Smoother, Predictor):
    """
    Class that implements the Kalman Filter as an iterative graph message passing routine.
    Currently just a Smoother.
    """

    def __init__(self, H, Q, R):
        """
        Initialises the graphical model.

        :param H: The measurement model 2d n x m tensor
        :param Q: The motion noise model 2d square tensor
        :param R: The measurement noise model 2d square tensor
        """
        super().__init__()
        self.H = H
        self.Q = Q
        self.R = R
        self.negQinv = -Q.inverse()
        self.Qinv = Q.inverse()
        self.HtRinv = H.t().matmul(R.inverse())

    def diff_past_curr(self, x_past, x_curr, Fs):
        """
        Calculates xt - Fxt-1.
        xcurr and xpast must have the same dimensions.
        should be dims x samples shape.
        :param x_past: xt-1
        :param x_curr: xt
        :param Fs: The F for each time step in the sequence
        :return:
        """
        res = x_curr - Fs.matmul(x_past)
        return res

    def diff_curr_fut(self, x_curr, x_fut, Fs):
        """
        Calculates xt+1 - Fxt
        Should be dims x samples shape.
        :param x_curr: xt
        :param x_fut: xt+1
        :param Fs: The F for each time step
        :return:
        """
        res = x_fut - Fs.matmul(x_curr)
        return res

    def diff_y_curr(self, ys, x_curr):
        """
        Calculates y - Hxt
        :param ys: y
        :param x_curr: xt
        :return:
        """
        res = ys - self.H.matmul(x_curr)
        return res

    def iterate(self, xs, ys, gamma: float, iterations: int = 200):
        """
        This will compute the graphical model solution to the problem.
        Use once() to call a single iteration.
        :param xs: estimates of the states
        :param ys: observations
        :param gamma: the factor by which to update the xs.
        :param iterations: The number of iterations the iterative process should do.
        :return:
        """
        # xs should be batch x feat x samples
        # ys should be batch x feat x samples.
        x = xs
        # iterate up to the number of iterations
        for i in range(iterations):
            # each time calculate the messages
            messages = self.forward(x, ys)
            # update the xs by the sum of messages * gamma
            x += sum(messages) * gamma

        # return the result.
        return x

    def forward(self, xs, ys, Fs):
        """
        Runs a single iteration of the graphical model.
        Returns the messages, xs and ys.
        xs should be (batch x feat x samples)
        ys should be (batch x feat x samples)
        :param xs: estimates of the state
        :param ys: observations.
        :param Fs: The F at each timestept in the sequence
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
        if ys.shape[2] != xs.shape[2]:
            ys = ys.permute(0, 2, 1)

        # result dims batch x feat x samples
        m1 = self.negQinv.matmul(self.diff_past_curr(x_past, xs, Fs))
        m2 = Fs.t().matmul(self.Qinv).matmul(self.diff_curr_fut(xs, x_future, Fs))
        m3 = self.HtRinv.matmul(self.diff_y_curr(ys, xs))

        # return messages
        return [m1, m2, m3]

    def predict(self, n: int, xs, ys, gamma: float, iterations: int = 200):
        """
        Predicts up to n time steps into the future.
        inputs expected to be (batch x feat x seq)
        :param n:
        :param xs:
        :param ys:
        :param gamma:
        :param iterations:
        :return:
        """
        # add n steps to the end of ys. (should be sequence dimension)
        ys = torch.nn.functional.pad(ys, (0, 0, 0, n), 'constant', 0.)

        return self.iterate(xs=xs, ys=ys, gamma=gamma, iterations=iterations)
