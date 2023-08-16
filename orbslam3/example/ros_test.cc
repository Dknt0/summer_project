#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip> // IO 流控制文件
#include <memory>

// OpenCV header file
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// ORB SLAM 3 main class
// #include "include/System.h"

// ROS header file
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

using namespace std;

std::string path_to_vocabulary = "~/Projects/uav_sim/src/orbslam3/Vocabulary/ORBvoc.txt";
std::string path_to_settings = "~/Projects/uav_sim/src/orbslam3/example/ros_test.yaml";

class ImageNode : public rclcpp::Node {
public:
    ImageNode() : Node("image_node") {
        // std::bind(&ImageNode::image_callback, this, std::placeholders::_1);
        std::bind(&ImageNode::test_cb, this, std::placeholders::_1);
        this->_image_listener = this->create_subscription<sensor_msgs::msg::Image>(
            "car_image",
            10,
            std::bind(&ImageNode::image_callback, this, std::placeholders::_1)
        );

        cv::namedWindow("test", cv::WINDOW_GUI_EXPANDED);

        // 创建系统
        // _slam = std::make_shared<ORB_SLAM3::System>(_path_to_vocabulary, _path_to_settings, ORB_SLAM3::System::MONOCULAR, true);
        // 获取比例
        // _imageScale = _slam->GetImageScale();

    }

private:
    void image_callback(const sensor_msgs::msg::Image &msg) const {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, "rgb8");
        cv::imshow("test", cv_ptr->image);
        char key = cv::waitKey(1);

    }
    void test_cb(const std_msgs::msg::String &msg) const { }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _image_listener;
    // std::shared_ptr<ORB_SLAM3::System> _slam;
    float _imageScale;

};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageNode>());
    rclcpp::shutdown();

    // if (argc < 3) {
    //     cerr << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
    //     return 1;
    // }

    // cv::VideoCapture cap;
    // cap.open(2, cv::CAP_ANY);

    // if (!cap.isOpened()) {
    //     cerr << "Failed to open camera" << endl;
    //     return 1;
    // }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // ORB_SLAM3::System SLAM(path_to_vocabulary, path_to_settings, ORB_SLAM3::System::MONOCULAR, true);
    // float imageScale = SLAM.GetImageScale(); // 比例
    

    // cv::namedWindow("test", cv::WINDOW_GUI_NORMAL);
    // char key;

    // cv::Mat img;

    // while (!SLAM.isShutDown()) {
    //     // cap.read(img);
    //     cv::imshow("test", img);

    //     int width = img.cols * imageScale;
    //     int height = img.rows * imageScale;
    //     cv::resize(img, img, cv::Size(width, height));
    //     chrono::seconds sec = chrono::duration_cast<chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch());
    //     // cout<<"current seconds: " << sec.count() << endl;
    //     SLAM.TrackMonocular(img, sec.count());

    //     key = cv::waitKey(1);
    //     if (key == 27) {
    //         break;
    //     }
    // }


    cv::destroyAllWindows();

    return 0;
}
