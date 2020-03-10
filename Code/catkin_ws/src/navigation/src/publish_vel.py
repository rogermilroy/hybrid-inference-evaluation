#! /usr/bin/env python

from queue import Queue

import rospy
from geometry_msgs.msg import Twist
from utils import empty_twist


class PublishVel:

    def __init__(self, velq: Queue):
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.rate = rospy.Rate(100)
        self.velq = velq

    def publish(self):
        cache_vel = empty_twist()

        while True:
            if self.velq.empty():
                # print("Empty Queue")
                vel_to_publish = cache_vel
            else:
                vel_to_publish = self.velq.get()
                cache_vel = vel_to_publish
            self.pub.publish(vel_to_publish)
            # print("Current vel publishing",vel_to_publish)
            self.rate.sleep()


def start_publisher(velq: Queue):
    """
    Wrapper to start publisher for threading.
    :return: None
    """
    # rospy.init_node('publish_vel')
    pub = PublishVel(velq=velq)
    pub.publish()
