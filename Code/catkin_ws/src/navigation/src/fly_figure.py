#! /usr/bin/env python3

import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import Pose
from navigation.msg import NavigateToPointAction, NavigateToPointGoal
from utils import enable_motors

if __name__ == '__main__':
    rospy.init_node("fly_figure")
    enable_motors()
    print("Motors Enabled")
    nav_client = SimpleActionClient('navigate_to_point', NavigateToPointAction)
    nav_client.wait_for_server()

    first_point = Pose()
    first_point.position.z = 5.
    first = NavigateToPointGoal(first_point)

    second_point = Pose()
    second_point.position.z = 5.
    second_point.position.x = 3.
    second = NavigateToPointGoal(second_point)

    nav_client.send_goal(first)
    nav_client.wait_for_result()

    rospy.sleep(rospy.Duration(secs=5))

    nav_client.send_goal(second)
    nav_client.wait_for_result()
    print("Finished")
