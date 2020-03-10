#! /usr/bin/env python3

from queue import Queue
from threading import Thread

import rospy
from controller import PID
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from publish_vel import start_publisher
from utils import enable_motors


class NavigateToPoint:

    def __init__(self):
        self.current_pos = None

        rospy.init_node('navigate_to_point')

        # create instruction queue
        self.vel_q = Queue(maxsize=2)

        # create target position
        self.target = Pose()
        self.target.position.z = 2.

        # create pid
        self.pid = PID(P=1.2, I=0.1, D=0.4, target=self.target)

    def position_callback(self, data: Odometry):
        self.current_pos = data.pose.pose
        print("POSE ++++++++++++")
        print(self.current_pos.position)
        # print('target ============ ')
        # print(self.target.position)
        self.pid.update(self.current_pos)
        out = self.pid.output()
        # print("command = ", out)
        self.vel_q.put(out)

    def start(self):
        # create subscriber to deal with pose changes
        rospy.Subscriber('/ground_truth/state', Odometry, self.position_callback)

        # create publish thread
        publish_thread = Thread(target=start_publisher(velq=self.vel_q), daemon=True)
        publish_thread.start()

        rospy.spin()


if __name__ == '__main__':
    print("started")
    enable_motors()
    print("Motors Enabled")
    NavigateToPoint().start()
