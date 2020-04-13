#! /usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Twist, Pose
from hector_uav_msgs.srv import EnableMotors


def empty_twist() -> Twist:
    """
    Creates and returns a zero initialised Twist.
    This represents linear and angular velocities.
    :return:
    """
    tw = Twist()
    tw.linear.x = 0.0
    tw.linear.y = 0.0
    tw.linear.z = 0.0
    tw.angular.x = 0.0
    tw.angular.y = 0.0
    tw.angular.z = 0.0
    return tw


def empty_pose() -> Pose:
    """
    Creates and returns a zero initialised Pose.
    This represents the position and orientation.
    :return:
    """
    pos = Pose()
    pos.position.x = 0.0
    pos.position.y = 0.0
    pos.position.z = 0.0
    # note orientation should be initialised as 0, 0, 0, 1 by default. TODO check.
    return pos


def get_pos_numpy(pose: Pose):
    ret = np.zeros(3)
    ret[0] = pose.position.x
    ret[1] = pose.position.y
    ret[2] = pose.position.z
    return ret


def get_linear_numpy(tw: Twist):
    return np.array([tw.linear.x, tw.linear.y, tw.linear.z])


def set_linear_twist_numpy(tw: Twist, arr: np.ndarray):
    tw.linear.x = arr[0]
    tw.linear.y = arr[1]
    tw.linear.z = arr[2]
    return tw


def get_quaternion_numpy(pos: Pose):
    quat = np.zeros(4)
    quat[0] = pos.orientation.w
    quat[1] = pos.orientation.x
    quat[2] = pos.orientation.y
    quat[3] = pos.orientation.z
    return quat


def enable_motors():
    rospy.wait_for_service("enable_motors")
    try:
        enable = rospy.ServiceProxy("enable_motors", EnableMotors)
        success = enable(True)
        print(success)
    except rospy.ServiceException as e:
        print(e)


########################################################################################################################
# The below code was originally written by Octavio del Ser Perez and provided with permission.
#
########################################################################################################################


def quaternion_product(q, r):
    """
    https://www.mathworks.com/help/aerotbx/ug/quatmultiply.html?w.mathworks.com
    https://www.mathworks.com/help/fusion/ref/quaternion.rotatepoint.html
    Output quaternion product quatprod has the form of
    n=qxr=n0+in1+jn2+kn3
    where
    n0=(r0q0-r1q1-r2q2-r3q3)
    n1=(r0q1+r1q0-r2q3+r3q2)
    n2=(r0q2+r1q3+r2q0-r3q1)
    n3=(r0q3-r1q2+r2q1+r3q0)

    :param q:
    :param r:
    :return:
    """
    return [r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
            r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
            r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
            r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]]


def rotate_point_by_quaternion(point: np.array, q):
    """
    https://www.mathworks.com/help/fusion/ref/quaternion.rotatepoint.html
    Converts point [x,y,z] to a quaternion:

    uq=0+xi+yj+zk

    Normalizes the quaternion, q:

    qn=qGa2+b2+c2+d2

    Applies the rotation:re-iterate

    vq=quqq*

    Converts the quaternion output, vq, back to R3
    :param point:
    :param q:
    :return:
    """
    # add the imaginary w variable to the point to convert it to a quaternion
    # the w value is 0 when the point is real.
    r = [0] + point.tolist()
    # calculate the conjugate
    q_conj = q * [1, -1, -1, -1]
    # quaternion is normalized already so no need to reiterate. only take the last 3 variables (ignore w)
    return quaternion_product(quaternion_product(q, r), q_conj)[1:]


def get_vector_to_point_quaternion(point_from: np.array, point_to: np.array, quaternion_reference: np.array):
    """
    vector to point from the zero frame of reference (0,0,0,0) to another frame of reference (w,x,y,z)
    :param point_from:
    :param point_to:
    :param quaternion_reference:
    :return:
    """
    vector_to_point = point_to - point_from
    quaternion = quaternion_reference * [-1, 1, 1, 1]
    vector_to_point = rotate_point_by_quaternion(vector_to_point, quaternion)
    return np.array(vector_to_point)
