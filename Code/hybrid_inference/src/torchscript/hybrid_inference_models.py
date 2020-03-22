import torch
from src.models.graphical_model import KalmanGraphicalModel, KalmanInputGraphicalModel, ExtendedKalmanGraphicalModel
from src.models.graphical_nn_model import KalmanGNN
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


class HybridInference(Smoother, Predictor):

    def __init__(self, F, H, Q, R, gamma: float = 1e-4):
        super(HybridInference, self).__init__()
        self.graph = KalmanGraphicalModel(F=F, H=H, Q=Q, R=R)
        self.gnn = KalmanGNN(h_dim=48, x_dim=F.shape[0], y_dim=R.shape[0])
        self.H = H
        self.gamma = gamma

    def forward(self, ys):
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

        :param ys: (batch x seq x feat)
        :param us: (batch x seq x feat)
        :param iterations:
        :return:
        """
        xs = self.H.t().matmul(torch.transpose(ys, 1, 2))

        hx, hy = self.gnn.initialise_hx_y(ys)

        iter_preds = []

        for i in range(200):
            # compute the graphical model messages
            # if us is not None:
            messages = self.graph(xs, ys)
            # else:
            #     messages = self.graph(xs, ys)
            # compute the hidden states and epsilon correction
            eps, hx = self.gnn(hy, hx, messages[0], messages[1], messages[2])
            # update the xs with messages and epsilon.
            xs = xs + self.gamma * (eps + torch.sum(torch.stack(messages), 0, True).squeeze(0))
            iter_preds.append(xs)
        # return the final estimate of the positions.
        return xs

    @torch.jit.export
    def predict(self, n: int, ys, us):
        """
        Predicts up to n time steps into the future.
        inputs expected to be (batch x feat x seq)
        :param n:
        :param ys: (
        :param us:
        :param iterations:
        :return:
        """
        # us must match size otherwise error
        if us.shape[2] != (ys.shape[2] + n):
            raise Exception("Inputs (us) must be provided to line up with predicted time.")
        else:
            # add n steps to the end of ys. (should be sequence dimension)
            ys = torch.nn.functional.pad(ys, (0, 0, 0, n), 'constant', 0.)

        return self.forward(ys=ys)


####################################################################################################
# Input version of HI
####################################################################################################


class HybridInferenceInput(Smoother, Predictor):

    def __init__(self, F, H, Q, R, G, gamma: float = 1e-4):
        super(HybridInferenceInput, self).__init__()
        self.graph = KalmanInputGraphicalModel(F=F, H=H, Q=Q, R=R, G=G)
        self.gnn = KalmanGNN(h_dim=48, x_dim=F.shape[0], y_dim=R.shape[0])
        self.H = H
        self.gamma = gamma

    def forward(self, ys, us):
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
        (batch x feat x seq)
        :param ys:
        :param us:
        :param iterations:
        :return:
        """
        xs = self.H.t().matmul(torch.transpose(ys, 1, 2))

        hx, hy = self.gnn.initialise_hx_y(ys)

        iter_preds = []

        for i in range(200):
            # compute the graphical model messages
            # if us is not None:
            messages = self.graph(xs, ys, us)
            # else:
            #     messages = self.graph(xs, ys)
            # compute the hidden states and epsilon correction
            eps, hx = self.gnn(hy, hx, messages[0], messages[1], messages[2])
            # update the xs with messages and epsilon.
            xs = xs + self.gamma * (eps + torch.sum(torch.stack(messages), 0, True).squeeze(0))
            iter_preds.append(xs)
        # return the final estimate of the positions.
        return xs, torch.stack(iter_preds)

    @torch.jit.export
    def predict(self, n: int, ys, us):
        """
        Predicts up to n time steps into the future.
        inputs expected to be (batch x feat x seq)
        :param n:
        :param ys: (
        :param us:
        :param iterations:
        :return:
        """
        # us must match size otherwise error
        if us.shape[2] != (ys.shape[2] + n):
            raise Exception("Inputs (us) must be provided to line up with predicted time.")
        else:
            # add n steps to the end of ys. (should be sequence dimension)
            ys = torch.nn.functional.pad(ys, (0, 0, 0, n), 'constant', 0.)

        return self.forward(ys=ys, us=us)


####################################################################################################
#  EKF version (takes F as input for each forward pass.
####################################################################################################


class ExtendedKalmanHybridInference(Smoother, Predictor):

    def __init__(self, H, Q, R, gamma: float = 1e-4):
        super(ExtendedKalmanHybridInference, self).__init__()
        self.graph = ExtendedKalmanGraphicalModel(H=H, Q=Q, R=R)
        self.gnn = KalmanGNN(h_dim=48, x_dim=Q.shape[0], y_dim=R.shape[0])
        self.H = H
        self.gamma = gamma

    def forward(self, ys, Fs):
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
        (batch x feat x seq)
        :param ys:
        :param Fs:
        :return:
        """
        xs = self.H.t().matmul(torch.transpose(ys, 1, 2))

        hx, hy = self.gnn.initialise_hx_y(ys)

        iter_preds = []

        for i in range(200):
            # compute the graphical model messages
            # if us is not None:
            messages = self.graph(xs, ys, Fs)
            # else:
            #     messages = self.graph(xs, ys)
            # compute the hidden states and epsilon correction
            eps, hx = self.gnn(hy, hx, messages[0], messages[1], messages[2])
            # update the xs with messages and epsilon.
            xs = xs + self.gamma * (eps + torch.sum(torch.stack(messages), 0, True).squeeze(0))
            iter_preds.append(xs)
        # return the final estimate of the positions.
        return xs, torch.stack(iter_preds)

    @torch.jit.export
    def predict(self, n: int, ys, Fs):
        """
        Predicts up to n time steps into the future.
        inputs expected to be (batch x feat x seq)
        :param n:
        :param ys: (
        :param Fs:
        :return:
        """

        # add n steps to the end of ys. (should be sequence dimension)
        ys = torch.nn.functional.pad(ys, (0, 0, 0, n), 'constant', 0.)

        return self.forward(ys=ys, Fs=Fs)
