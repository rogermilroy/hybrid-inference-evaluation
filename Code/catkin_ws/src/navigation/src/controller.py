#! /usr/bin/env python3

from abc import ABC, abstractmethod

import numpy as np
import rospy
from geometry_msgs.msg import Twist, Pose
from utils import empty_twist, empty_pose, get_pos_numpy, set_linear_twist_numpy, get_vector_to_point_quaternion, \
    get_quaternion_numpy


class Controller(ABC):
    """
    Controller class that provides an abstraction for controllers.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, current):
        pass

    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def set_target(self, target):
        pass


class PID(Controller):
    """
    PID comtroller for navigating to points.
    TODO think about how hard it would be to convert this all to C++. Iteractions with other code would be harder/slower.
    """

    def __init__(self, P: float, I: float, D: float, target: Pose = None):
        """
        Initialise all the variables necessary.
        :param P: The P (proportional) coefficient.
        :param I: The I (integral) coefficient.
        :param D: The D (differential) coefficient.
        """
        self.P = P
        self.I = I
        self.D = D
        self._time = rospy.get_rostime().nsecs / 1e6  # convert to ms
        self._sum = np.zeros(3)
        self._target: Pose = target if target is not None else empty_pose()
        self._output: Twist = empty_twist()
        self.average_window: float = 500.
        super().__init__()

    def set_target(self, target: Pose) -> None:
        """
        Sets a new target position.
        :param target: The new desired position.
        :return:
        """
        self._target = target

    def update(self, current: Pose) -> None:
        # get times
        now = rospy.get_rostime().nsecs / 1e6  # convert to ms
        time_delta = now - self._time
        self._time = now
        if time_delta == 0.:
            return
        # print("TIme delta", time_delta)

        # compute the difference (error)
        prop: np.ndarray = get_vector_to_point_quaternion(get_pos_numpy(current), get_pos_numpy(self._target),
                                                          get_quaternion_numpy(current))
        print("Error ", prop)

        # compute the differential
        diff: np.ndarray = prop / time_delta

        # compute the integral ( average over a window )
        self._sum = (self.average_window - time_delta) * (self._sum / self.average_window)
        self._sum = time_delta * (prop / self.average_window)

        # set the output.
        out_nump = (self.P * prop) + (self.I * self._sum) + (self.D * diff)
        print('out Numpy', out_nump)
        self._output = set_linear_twist_numpy(self._output, out_nump)

    def output(self) -> Twist:
        return self._output
