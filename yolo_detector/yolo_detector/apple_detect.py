#!/usr/bin/python3
"""
Gazebo + Yolov5 detector
Author: Dknt
Date: 2022.12
Reference: https://blog.csdn.net/Silghz/article/details/123959805
"""

"""
Calibration parameters
---
Point
img: x1 x2 y1 y2
joint: x y
+++
Point1
img: 642 710 170 246
joint: 0.2 0.255
+++
Point2
img: 430 505 140 216
joint: 0.368 0.415

612 344 680 418


494 93 566 168

# 0.196  0.252
# 381 646
# 0.41  0.355
# 130.5 530 
"""


import cv2 as cv
import detect

import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge , CvBridgeError
import sys

a = detect.detectapi(weights='yolov5s.pt')

# ax = -0.000805755
# bx = 0.71769038
# # ay = -0.0054
# ay = -0.000805755
# by = 0.5610

ax = -0.0008542914171656685
bx = 0.5214850299401197 + 0.020
# ay = -0.0054
ay = -0.0008542914171656685
by = 0.825603448275862

def detection(data):
    result,names =a.detect([data])
    img=result[0][0] # 第一张图片的处理结果图片

    for cls,(x1,y1,x2,y2),conf in result[0][1]: # 第一张图片的处理结果标签
        if cls == 47:  # 只检测苹果
            print(cls,x1,y1,x2,y2,conf)
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
            cv.putText(img,names[cls],(x1,y1-20),cv.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
            x_img = (x2 + x1) / 2
            y_img = (y2 + y1) / 2
            x_apple = ax * y_img + bx
            y_apple = ay * x_img + by
            rospy.set_param('pos_apple', [x_apple, y_apple])

    cv.imshow("video",img)
    cv.waitKey(1)

def callback(data):
    src = bridge.imgmsg_to_cv2(data, 'bgr8')
    detection(src)

if __name__ == '__main__':
    rospy.init_node('detector')
    rospy.Subscriber('/agrolabCamera/image_raw', Image, callback)
    bridge = CvBridge()
    rospy.spin()
