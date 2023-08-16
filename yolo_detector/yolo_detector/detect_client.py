#!/usr/bin/python3

'''
This is a part of code for my summer internship project 2023 in BMSTU.
YOLO 目标检测服务

Dknt 2023.8
'''

# import sys
# sys.path.append("/home/dknt/Projects/uav_sim/src/yolo_detector/yolo_detector")

import rclpy
from rclpy.node import Node
from ofb_ctr.srv import TargetDetection
from sensor_msgs.msg import Image
from cv_bridge import CvBridge , CvBridgeError


import cv2 as cv

# import detect  # YOLOv5 detector

bridge = CvBridge()  # 图片转换
path_to_img = "/home/dknt/Desktop/cat.jpg"

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('yolo_client')
        self.cli = self.create_client(TargetDetection, 'yolo_detection')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = TargetDetection.Request()

    def send_request(self):
        img = cv.imread(path_to_img)
        self.req.src = bridge.cv2_to_imgmsg(img)
        # cv.imshow("test", img)
        # cv.waitKey(0)


        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

        res = bridge.imgmsg_to_cv2(self.future.result().res)
        cv.imshow("test", res)
        cv.waitKey(0)
        return self.future.result()


def main():
    rclpy.init()

    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request()
    minimal_client.get_logger().info(
        'Finished.')

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
