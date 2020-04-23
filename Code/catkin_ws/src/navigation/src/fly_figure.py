#! /usr/bin/env python3

import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import Pose
from navigation.msg import NavigateToPointAction, NavigateToPointGoal
from utils import enable_motors


def square() -> list:
    """
    Creates a sequence of instructions to fly in a square, horizontally.
    :return:
    """
    goals = list()

    first_point = Pose()
    first_point.position.z = 4.
    goals.append(NavigateToPointGoal(first_point))

    second_point = Pose()
    second_point.position.z = 4.
    second_point.position.x = 2.
    goals.append(NavigateToPointGoal(second_point))

    third_point = Pose()
    third_point.position.z = 4.
    third_point.position.x = 2.
    third_point.position.y = 2.
    goals.append(NavigateToPointGoal(third_point))

    fourth_point = Pose()
    fourth_point.position.z = 4.
    fourth_point.position.y = 2.
    goals.append(NavigateToPointGoal(fourth_point))

    final_point = Pose()
    final_point.position.z = 4.
    goals.append(NavigateToPointGoal(final_point))
    return goals


if __name__ == '__main__':
    rospy.init_node("fly_figure")
    enable_motors()
    print("Motors Enabled")
    nav_client = SimpleActionClient('navigate_to_point', NavigateToPointAction)
    nav_client.wait_for_server()

    for i in range(4):
        goals = square()

        for goal in goals:
            print("Sending")
            nav_client.send_goal(goal)
            print("Waiting")
            nav_client.wait_for_result()
            print("Pausing")
            rospy.sleep(rospy.Duration(secs=5))

    print("Finished")
