/**
 * A cpp test for Python detection server.
 * 
 * Dknt 2023.8
*/

#include <iostream>
#include <rclcpp/rclcpp.hpp>

// OpenCV header file
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// CV bridge
#include "cv_bridge/cv_bridge.h"

#include <ofb_ctr/srv/target_detection.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

std::string path_to_img = "/home/dknt/Desktop/cat.jpg";

class TestNode : public rclcpp::Node {
public:
    TestNode() : Node("detect_client_test") {
        
        // 创建客户端
        _client = this->create_client<ofb_ctr::srv::TargetDetection>("yolo_detection");

        // 等待服务
        while (!_client->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
                exit(1); // 退出
            }
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not avaible, waiting again...");
        }
        
    }
    rclcpp::Client<ofb_ctr::srv::TargetDetection>::SharedPtr _client;

private:
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    cv::Mat src = cv::imread(path_to_img);
    // 创建请求
    // 这里犯过给空共享指针所指的对象赋值的错误
    assert(path_to_img.data() != nullptr && "Failed to open image");
    cv_bridge::CvImage cv_img(std_msgs::msg::Header(), "bgr8", src);

    auto node = std::make_shared<TestNode>();

    ofb_ctr::srv::TargetDetection::Request request;
    auto msg = (cv_img.toImageMsg());
    
    std::cout << "test" << std::endl;
    request.src.height = msg->height;

    request.src = *msg; // cv::Mat 转消息
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Created request");
    
    auto request_qpt = std::make_shared<ofb_ctr::srv::TargetDetection::Request>(request);
    
    auto res_future = node->_client->async_send_request(request_qpt);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Sent a request.");
    
    // 很神奇，这个必须放到外面才行...
    if (rclcpp::spin_until_future_complete(node, res_future) == rclcpp::FutureReturnCode::SUCCESS) {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Received result.");
    }
    else {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed.");
    }


    auto res = res_future.get();
    cv_bridge::CvImagePtr cv_ptr_res;
    cv_ptr_res = cv_bridge::toCvCopy(res->res, "bgr8");
    
    // 显示结果
    cv::imshow("result", cv_ptr_res->image);
    cv::waitKey(0);


    // rclcpp::spin(std::make_shared<TestNode>());
    rclcpp::shutdown();

    return 0;
}
