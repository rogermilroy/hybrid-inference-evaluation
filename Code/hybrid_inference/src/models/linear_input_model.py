from torch import tensor, eye
import torch
from torch.distributions import MultivariateNormal


class ConstantVelocityWInputModel:

    def __init__(self, x0, input_fn = None, T: float = 1., lambdasq: float = 0.5 ** 2,
                 sigma_x: float = 0.15 ** 2, sigma_y: float = 0.15 ** 2):
        """
        Sets the parameters of the linear model. State is encoded [ x, vx, y, vy].T
        :param: x0: tensor The initial state. Should be 1d tensor of size 4.
        :param T: Time
        :param lambdasq: noise coefficient
        :param sigma_x: Gaussian noise sd
        :param sigma_y: Gaussian noise sd
        """
        self.x0 = x0
        self.x = self.x0
        # transition matrix
        self.A = tensor([[1., T, 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., T],
                         [0., 0., 0., 1.]])

        # transition noise
        transition_covariance = tensor([[T * sigma_x, 0., 0., 0.],
                                        [0., T * sigma_x, 0., 0.],
                                        [0., 0., T * sigma_y, 0.],
                                        [0., 0., 0., T * sigma_y]])

        transition_mean = torch.zeros(self.A.size()[0])
        self.Q = MultivariateNormal(transition_mean, transition_covariance)

        # measurement projection matrix (extracts position only)
        self.H = tensor([[1., 0., 0., 0.],
                         [0., 0., 1., 0.]])

        # input gain matrix. input is a 2 vector of [ax, ay]
        self.G = tensor([[(T ** 2) / 2, T, 0., 0.],
                         [0., 0., (T ** 2) / 2, T]]).t()

        # measurement noise.
        measurement_covariance = lambdasq * eye(2)
        measurement_mean = torch.zeros(measurement_covariance.size()[0])
        self.R = MultivariateNormal(measurement_mean, measurement_covariance)
        self.t = 0.
        if input_fn is None:
            self.input_function = self.input_fn

    def input_fn(self, t):
        """
        Generates inputs (either randomly or from some function).
        Will be acceleration amounts. Cosine and sine for x and y. should draw a circle ish.
        :return:
        """
        return torch.cat([torch.sin(torch.tensor([t/30])), torch.cos(torch.tensor([t/30]))])

    def __call__(self, *args, **kwargs):
        self.x = (self.A @ self.x.t()) + (self.G @ self.input_fn(self.t)) + self.Q.sample()
        self.measurement = (self.H @ self.x.t()) + self.R.sample()
        self.t += 1.
        return self.x, self.measurement

