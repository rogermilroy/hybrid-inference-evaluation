#! /usr/bin/env python3

from queue import Queue, Full
from threading import Thread

import rospy
from actionlib import SimpleActionServer
from controller import PID
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from navigation.msg import NavigateToPointAction, NavigateToPointFeedback, NavigateToPointActionGoal
from publish_vel import start_publisher
from utils import get_linear_numpy


class NavigateToPoint:

    def __init__(self):
        self._current_pos = None

        # create instruction queue
        self._vel_q = Queue(maxsize=2)

        # create target position
        self._target = Pose()
        self._target.position.z = 0.

        # create pid
        self._pid = PID(P=1., I=0.2, D=0.4, target=self._target)

        self._action_server = SimpleActionServer('navigate_to_point', NavigateToPointAction, self.set_target_callback,
                                                 auto_start=False)
        self._goal = None

    def position_callback(self, data: Odometry):
        self._current_pos = data.pose.pose
        print("POSE ++++++++++++")
        print(self._current_pos.position)
        # print('target ============ ')
        # print(self.target.position)
        self._pid.update(self._current_pos)
        out = self._pid.output()
        # print("command = ", out)
        try:
            self._vel_q.put(out, block=False)
        except Full as e:
            print(e)
        if self._goal:
            if sum(get_linear_numpy(out)) < 0.1:
                self._action_server.set_succeeded()
            else:
                # publish feedback
                self._feedback = NavigateToPointFeedback(out)
                self._action_server.publish_feedback(self._feedback)

    def set_target_callback(self, goal: NavigateToPointActionGoal):
        self._target = goal.pose
        self._pid.set_target(self._target)

    def start(self):
        # create subscriber to deal with pose changes
        # rospy.Subscriber('/ground_truth/state', Odometry, self.position_callback)
        rospy.Subscriber("/state", Odometry, self.position_callback)

        self._action_server.start()

        # create publish thread
        publish_thread = Thread(target=start_publisher(velq=self._vel_q), daemon=True)
        publish_thread.start()

        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('navigate_to_point')
    NavigateToPoint().start()
