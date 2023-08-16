#!/usr/bin/python3
"""
Author: Dknt
Date: 2022.11
"""
import rospy
import numpy as np
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
length = 0.43
width = 0.54
height = 0.4
period = 5

def move(x,y,z):
    rospy.init_node('agrolab_move')
    base_link_to_X = rospy.Publisher('/agrolab/base_link_to_X_controller/command', Float64, queue_size = 100)
    X_to_Y = rospy.Publisher('/agrolab/X_to_Y_controller/command', Float64, queue_size = 100)
    Y_to_Z = rospy.Publisher('/agrolab/Y_to_Z_controller/command', Float64, queue_size = 100)

    for i in range(10):
        base_link_to_X.publish(x)
        X_to_Y.publish(y)
        Y_to_Z.publish(z) 
        rospy.sleep(3)
    for i in range(10):
        Y_to_Z.publish(0)
    # while not rospy.is_shutdown():
        # t = rospy.get_time()
        # k = t % period
        # t = period / 2
        # if k < period * 0.5:
        #     base_link_to_X.publish((k) / t * length)
        #     X_to_Y.publish((k) / t * width)
        #     Y_to_Z.publish((k) / t * height)
        # else:
        #     base_link_to_X.publish(length - (k - t) / t * length)
        #     X_to_Y.publish(width - (k - t) / t * width)
        #     Y_to_Z.publish(height - (k - t) / t * height)



if __name__ == '__main__':
    position = input("Please input X,Y,Z:") 

    move(position)