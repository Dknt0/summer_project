/**
 * 工具程序 从ROS中录制SLAM数据集 输出格式参考KITTI 带Groundtruth
 * 
 * dknt 2023.7.29
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> // IO 流控制
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>

// OpenCV header file
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// ROS header file
#include "rclcpp/rclcpp.hpp"
// #include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

// CV bridge
#include "cv_bridge/cv_bridge.h"

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::string;

string path_to_dataset = "/home/dknt/Datasets/uav_sim_rgbd/00";

/**
 * 数据记录类
 * 
 * 保存图片到数据集目录，同时创建对应轨迹文件，格式参考 KITTI
 * 为保证图像与里程计信息一致，开单独线程保存磁盘
 * 频率25
*/
class RecorderNode : public rclcpp::Node {
public:
    RecorderNode() : Node("recorder_node") {
        this->_counter = 0;
        this->_ofs_times.open(path_to_dataset + "/times.txt", std::ios::out | std::ios::trunc);
        this->_ofs_gd.open(path_to_dataset + "/groundtruth.txt", std::ios::out | std::ios::trunc);
        this->_prefix_color = path_to_dataset + "/image_0/";
        this->_prefix_depth = path_to_dataset + "/depth/";

        if (!this->_ofs_times.is_open() && this->_ofs_gd.is_open()) {
            std::cerr << "Failed to open file" << std::endl;
        }

        // 创建收听者
        this->_rgb_image_sub = this->create_subscription<sensor_msgs::msg::Image>("camera", 10,
                                    std::bind(&RecorderNode::rgb_image_callback, this, _1));
        this->_depth_image_sub = this->create_subscription<sensor_msgs::msg::Image>("depth_camera", 10,
                                    std::bind(&RecorderNode::depth_image_callback, this, _1));
        this->_pose_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>("x500_pose", 10,
                                    std::bind(&RecorderNode::pose_callbask, this, _1));
        this->_timer = this->create_wall_timer(40ms, std::bind(&RecorderNode::timer_callback, this)); // 25Hz        
    }

    ~RecorderNode() {
        this->_ofs_times.close();
        this->_ofs_gd.close();
        std::cout << "Recorded" << std::endl;
    }

private:
    // 彩色图像话题回调函数
    void rgb_image_callback(const sensor_msgs::msg::Image &msg) const {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        this->_color_img_mutex.lock();
        this->_color_img = cv_ptr->image;
        // cv::imshow("rgb_viewer", this->_color_img);
        this->_color_img_mutex.unlock();
        // cv::waitKey(1);
    }

    // 深度图像话题回调函数
    void depth_image_callback(const sensor_msgs::msg::Image &msg) const {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        this->_depth_img_mutex.lock();
        this->_depth_img = cv_ptr->image;
        // cv::imshow("depth_viewer", this->_depth_img);
        this->_depth_img_mutex.unlock();
        // cv::waitKey(1);
    }

    // 里程计话题回调函数
    void pose_callbask(const geometry_msgs::msg::PoseStamped &msg) const {
        this->_pose_mutex.lock();
        this->_pose = msg;
        this->_pose_mutex.unlock();
    }

    // 定时器回调函数
    void timer_callback() {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << this->_counter; // 图片名为序号，用0补全
        double time = _pose.header.stamp.sec + _pose.header.stamp.nanosec * 1e-9;

        // 存图
        _color_img_mutex.lock();
        _depth_img_mutex.lock();
        _pose_mutex.lock();
        if (this->_color_img.data != nullptr) {
            cv::imwrite(this->_prefix_color + ss.str() + ".png", this->_color_img);
            // cv::imwrite(this->_prefix_depth + ss.str() + ".tiff", this->_depth_img);
            this->_ofs_times << time << std::endl;
            this->_ofs_gd << time << ' ' 
                        << _pose.pose.position.x << ' '
                        << _pose.pose.position.y << ' '
                        << _pose.pose.position.z << ' '
                        << _pose.pose.orientation.x << ' '
                        << _pose.pose.orientation.y << ' '
                        << _pose.pose.orientation.z << ' '
                        << _pose.pose.orientation.w << ' ' << std::endl;
            _counter++;
        }
        else {
            RCLCPP_ERROR(this->get_logger(), "Failed to save image.");
        }
        _color_img_mutex.unlock();
        _depth_img_mutex.unlock();
        _pose_mutex.unlock();
    }

    // 话题收听者
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _rgb_image_sub; // 彩色图像收听者
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _depth_image_sub; // 深度图像收听者
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _pose_sub; // 位姿收听者
    rclcpp::TimerBase::SharedPtr _timer; // 定时器

    // 当前值
    mutable cv::Mat _color_img;
    mutable cv::Mat _depth_img;
    mutable geometry_msgs::msg::PoseStamped _pose;

    // 互斥锁
    mutable std::mutex _color_img_mutex;
    mutable std::mutex _depth_img_mutex;
    mutable std::mutex _pose_mutex;

    int _counter;
    std::fstream _ofs_times;
    std::fstream _ofs_gd;
    std::string _prefix_color;
    std::string _prefix_depth;

};

int main(int argc, char **argv) {

    if (argc > 1) {
        path_to_dataset = argv[1];
        std::cout << "Save dataset to path: " << path_to_dataset << std::endl;
    }


    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RecorderNode>());
    rclcpp::shutdown();

    return 0;
}
