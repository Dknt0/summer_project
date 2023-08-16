#!/usr/bin/python3
"""
Gazebo + Yolov5 detector
Date: 2022.12
Reference: https://blog.csdn.net/Silghz/article/details/123959805
"""

import cv2 as cv
import detect
import time

import numpy as np
import rospy
from move1 import move
from sensor_msgs.msg import Image
from cv_bridge import CvBridge , CvBridgeError
import sys

a = detect.detectapi(weights='yolov5s.pt')
flag1 = 1

def detection(data):
    global flag1
    result,names =a.detect([data])
    img=result[0][0] # 第一张图片的处理结果图片
    position_x = 0
    position_y = 0
    position_z = 0
    for cls,(x1,y1,x2,y2),conf in result[0][1]: # 第一张图片的处理结果标签
        if cls == 63:
            print(cls,x1,y1,x2,y2,conf)
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
            cv.putText(img,names[cls],(x1,y1-20),cv.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
            position_x = (1034-x1)*0.54/780
            position_y = (481-y1)*0.43/457
            position_z = 0
    # t2 = time.time()
    # if flag1 == 1:
    #     t1 = time.time()
    #     xc=(x1+x2)*0.5
    #     flag1 = 0
        
    #     if t2-t1>=3 and abs(xc-(x1+x2)*0.5)<10:
        move(position_x,position_y,0.0)
        flag1 = 1
            #254 25
            #1034 24
            #255 481
            #1034 479 bl_to_X = 0.11
            # 1280 720

    cv.imshow("vedio",img)
    cv.waitKey(1)

def callback(data):
    src = bridge.imgmsg_to_cv2(data, 'bgr8')
    detection(src)

if __name__ == '__main__':
    rospy.init_node('detector')
    rospy.Subscriber('/agrolabCamera/image_raw', Image, callback)
    bridge = CvBridge()
    rospy.spin()
