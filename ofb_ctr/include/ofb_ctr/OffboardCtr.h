/**
 * Header file of Offboard control class
 * 
 * This is a part of code for my summer internship project 2023 in BMSTU.
 * 
 * Dknt 2023.7
*/

#ifndef OFFBOARDCTR_H
#define OFFBOARDCTR_H

#include <iostream>

// For multi-thread
#include <chrono>
#include <thread>
#include <future>
#include <mutex>

// MAVSDK header files, using plugins Telemetry, Action, Offboard
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/log_callback.h>

// For keyboard control
#include <unistd.h>
#include <termios.h>

// ROS 2
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <ofb_ctr/srv/target_detection.hpp>

// CV bridge
#include <cv_bridge/cv_bridge.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// UBUNTU/LINUX terminal color codes
#define RESET       "\033[0m"
#define BLACK       "\033[30m"          /* Black */
#define RED         "\033[31m"          /* Red */
#define GREEN       "\033[32m"          /* Green */
#define YELLOW      "\033[33m"          /* Yellow */
#define BLUE        "\033[34m"          /* Blue */
#define MAGENTA     "\033[35m"          /* Magenta */
#define CYAN        "\033[36m"          /* Cyan */
#define WHITE       "\033[37m"          /* White */
#define BOLDBLACK   "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"   /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"   /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"   /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"   /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"   /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"   /* Bold White */

// default velocity for keyboard control
#define VELOCITY_FORWARD 1.0f
#define VELOCITY_RIGHT 1.0f
#define VELOCITY_DOWN 1.0f
#define VELOCITY_YAW 35.0f
#define DATA_UPDATE_RATE 100.0f

// RGB Camera parameters
#define CAMERA_fx 615.9603271484375l
#define CAMERA_fy 616.227294921875l
#define CAMERA_cx 419.83026123046875l
#define CAMERA_cy 245.14314270019531l
#define CAMERA_width 848
#define CAMERA_hight 480

// ROS Gamble Control Node
class ControlNode : public rclcpp::Node {
public:
    ControlNode();
    void pubEuler(double roll, double pitch, double yaw);

private:
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr _rotor1_publisher;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr _rotor2_publisher;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr _rotor3_publisher;
};


// Main control class
class OffboardCtr {
public:
    OffboardCtr() { }

    /* 常用动作函数 */
    bool init(std::string &port); // 初始化
    bool arm();
    bool disarm();
    bool takeoff(float altitude);
    bool land();
    bool flyToPoint(); // ...

    /* 键盘控制函数 */
    void keyboardControl(); // 键盘控制
    void keyboardAccControl(); // 键盘加速度控制

    /* 云台控制函数 */
    void gimbalInit();
    void gimbalControl(const double yaw_tar, const double pitch_tar);
    void gimbalBalance(const double pitch_tar = 0.0);
    void objectTracking();

    // test function
    void telemetry_test();

private:
    void p_getKey(std::shared_ptr<mavsdk::Offboard::VelocityBodyYawspeed> velSetpoint,
                  std::shared_ptr<std::mutex> m_velSetpoint,
                  std::shared_ptr<std::promise<bool>> exit_prom);

    std::string _port;

    // MAVSDK interfaces
    std::shared_ptr<mavsdk::Mavsdk> _mavsdk;
    std::shared_ptr<mavsdk::System> _system;
    std::shared_ptr<mavsdk::Action> _action;
    std::shared_ptr<mavsdk::Telemetry> _telemetry;
    std::shared_ptr<mavsdk::Offboard> _offboard;

    // UAV States
    mavsdk::Telemetry::Position _position;
    mavsdk::Telemetry::VelocityNed _velocityNed;
    mavsdk::Telemetry::Odometry _odometry;
    
    // ROS Nodes
    std::shared_ptr<ControlNode> _controlNode;
};

char getch();

#endif
