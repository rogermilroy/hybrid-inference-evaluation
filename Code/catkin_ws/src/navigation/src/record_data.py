#! /usr/bin/env python3

import rospy
import torch
from geometry_msgs.msg import Quaternion, Point, Vector3
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation

i = 0


def odom_to_tensor(odom: Odometry) -> torch.tensor:
    return torch.cat([quaternion_to_eul_tensor(odom.pose.pose.orientation),
                      position_to_tensor(odom.pose.pose.position),
                      linear_to_tensor(odom.twist.twist.linear),
                      angular_to_tensor(odom.twist.twist.angular),
                      torch.zeros(3)])


def quaternion_to_euler(quat: Quaternion) -> list:
    r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    return r.as_euler('xyz').tolist()  # need to check about intrinsic vs extrinsic... this is extrinsic Euler angles.


def quaternion_to_eul_tensor(quat: Quaternion) -> torch.tensor:
    return torch.tensor(quaternion_to_euler(quat))


def position_to_tensor(pos: Point) -> torch.tensor:
    return torch.tensor([pos.x, pos.y, pos.z])


def vector3_to_tensor(vec: Vector3) -> torch.tensor:
    return torch.tensor([vec.x, vec.y, vec.z])


def linear_to_tensor(linear_vel: Vector3) -> torch.tensor:
    return vector3_to_tensor(linear_vel)


def angular_to_tensor(angular_vel: Vector3) -> torch.tensor:
    return vector3_to_tensor(angular_vel)


def record_twist_callback(data: Odometry):
    global i
    print("allback")
    print(int(data.header.stamp.nsecs) // 1000)
    print(odom_to_tensor(data))
    torch.save([torch.tensor(int(data.header.stamp.nsecs) // 1000), odom_to_tensor(data)],
               "recording1/odom-" + str(i) + ".pt")
    i += 1


if __name__ == '__main__':
    rospy.init_node("recorder")
    print("init node")
    sub = rospy.Subscriber("/ground_truth/state", Odometry, record_twist_callback)
    print('register callback')
    rospy.spin()
