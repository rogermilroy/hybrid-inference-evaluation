#! /usr/bin//env python

import rospy
from geometry_msgs import Twist
from queue import SimpleQueue


class PublishVel:

    def __init__(self, velq: SimpleQueue):
        self.pub = rospy.Publisher('/cmd_vel', Twist)
        self.rate = rospy.Rate(20)
        self.velq = velq

    @staticmethod
    def empty_twist() -> Twist:
        tw = Twist()
        tw.linear.x = 0.0
        tw.linear.y = 0.0
        tw.linear.z = 0.0
        tw.angular.x = 0.0
        tw.angular.y = 0.0
        tw.angular.z = 0.0
        return tw

    def publish(self):
        cache_vel = empty_twist()

        while True:
            if self.velq.empty():
                vel_to_publish = cache_vel
            else:
                vel_to_publish = self.velq.get()
                cache_vel = vel_to_publish
            self.pub.publish(vel_to_publish)
            self.rate.sleep()


def start_publisher(velq: SimpleQueue):
    """
    Wrapper to start publisher for threading.
    :return: None
    """
    pub = PublishVel(velq=velq)
    pub.publish()
