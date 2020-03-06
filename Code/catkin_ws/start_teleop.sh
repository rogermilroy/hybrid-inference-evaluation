#! /bin/bash

rosservice call /enable_motors "enable_motors: true";
rosrun teleop_twist_keyboard teleop_twist_keyboard.py;
