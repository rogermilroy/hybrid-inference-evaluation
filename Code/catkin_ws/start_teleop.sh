#! /bin/bash

rosservice call /enable_motors "enable: true";
rosrun teleop_twist_keyboard teleop_twist_keyboard.py;
