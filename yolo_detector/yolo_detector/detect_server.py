#!/usr/bin/python3

'''
This is a part of code for my summer internship project 2023 in BMSTU.
YOLO 目标检测服务

Dknt 2023.8
'''

import sys

sys.path.append("/home/dknt/Projects/uav_sim/src/yolo_detector/yolo_detector")

import rclpy
from rclpy.node import Node
from ofb_ctr.srv import TargetDetection
from sensor_msgs.msg import Image
from cv_bridge import CvBridge , CvBridgeError


import cv2 as cv

import detect  # YOLOv5 detector



a = detect.detectapi(weights='/home/dknt/Projects/uav_sim/src/yolo_detector/yolo_detector/yolov5s.pt')  # 检测器
bridge = CvBridge()  # 图片转换


class DetectServer(Node):

    def __init__(self):
        super().__init__('yolo_detector')
        # 创建服务
        self.srv = self.create_service(TargetDetection, 'yolo_detection', self.callback)
        print('Started.')

    def callback(self, request, response):
        src = bridge.imgmsg_to_cv2(request.src)
        cv.cvtColor(src, cv.COLOR_RGB2BGR, src)
        # response.sum = request.a + request.b
        result, names = a.detect([src])
        res = result[0][0] #第一张图片的处理结果图片
        u = 0
        v = 0
        
        # 我们只对车进行检测
        for cls,(x1,y1,x2,y2),conf in result[0][1]: #第一张图片的处理结果标签。
            # print(cls,x1,y1,x2,y2,conf)
            # cv.rectangle(res,(x1,y1),(x2,y2),(0,255,0))
            # cv.putText(res,names[cls],(x1,y1-20),cv.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
            if cls == 2:
                cv.rectangle(res,(x1,y1),(x2,y2),(0,255,0))
                cv.putText(res,names[cls],(x1,y1-5),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv.putText(res,str(round(conf,2)),(x1+40,y1-5),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                u = (int)((x1 + x2) / 2)
                v = (int)((y1 + y2) / 2)
                
                break # 只检测一个目标

        response.res = bridge.cv2_to_imgmsg(res)
        response.u = u
        response.v = v
        # cv.imshow('Server result', res)
        # cv.waitKey(1)

        # self.get_logger().info('Returned result')
        return response


def main():

    
    rclpy.init()
    minimal_service = DetectServer()
    rclpy.spin(minimal_service)
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()

