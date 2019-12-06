from torch import tensor, eye
import torch
from torch.distributions import MultivariateNormal


class ConstantVelocityModel:

    def __init__(self, x0, T: float = 1., lambdasq: float = 0.5 ** 2, sigma_x: float = 0.15 ** 2, sigma_y: float = 0.15 ** 2):
        """
        Sets the parameters of the linear model. State is encoded [ x, vx, y, vy ].T
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

        # measurement noise.
        measurement_covariance = lambdasq * eye(2)
        measurement_mean = torch.zeros(measurement_covariance.size()[0])
        self.R = MultivariateNormal(measurement_mean, measurement_covariance)

    def __call__(self, *args, **kwargs):
        self.x = (self.A @ self.x.T) + self.Q.sample()
        self.measurement = (self.H @ self.x.T) + self.R.sample()
        return self.x, self.measurement

