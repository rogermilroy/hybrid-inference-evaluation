from filterpy.kalman import ExtendedKalmanFilter
from numpy import dot


class QuadcopterExtendedKalman(ExtendedKalmanFilter):

    def __init__(self, dim_x, dim_y, Fs):
        super().__init__(dim_x, dim_y)
        self.i = 0
        self.Fs = Fs

    def predict_x(self, u=0):
        """
        Override standard predict x to use predefined Fs.
        :param u:
        :return:
        """
        self.F = self.Fs[self.i]
        self.i += 1
        super().predict_x(u=u)

