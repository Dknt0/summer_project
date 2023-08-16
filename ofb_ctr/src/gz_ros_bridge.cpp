/**
 * 话题转发节点
 * 转发彩色图像、深度图像、里程计信息
 * 
 * Dknt 2023.7.11
*/

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <ros_gz_bridge/ros_gz_bridge.hpp>

using RosGzBridge = ros_gz_bridge::RosGzBridge;

//////////////////////////////////////////////////
int main(int argc, char * argv[]) {
    // ROS node
    rclcpp::init(argc, argv);
    auto bridge_node = std::make_shared<RosGzBridge>(rclcpp::NodeOptions());

    // Set lazy subscriber on a global basis
    bool lazy_subscription = false;
    bridge_node->declare_parameter<bool>("lazy", false);
    bridge_node->get_parameter("lazy", lazy_subscription);

    // rgb
    ros_gz_bridge::BridgeConfig config1;
    config1.ros_topic_name = "camera";
    config1.gz_topic_name = "/uav_camera";
    config1.ros_type_name = "sensor_msgs/msg/Image";
    config1.gz_type_name = "gz.msgs.Image";
    config1.is_lazy = lazy_subscription;

    // depth
    ros_gz_bridge::BridgeConfig config2;
    config2.ros_topic_name = "depth_camera";
    config2.gz_topic_name = "/uav_depth_camera";
    config2.ros_type_name = "sensor_msgs/msg/Image";
    config2.gz_type_name = "gz.msgs.Image";
    config2.is_lazy = lazy_subscription;

    // odometry
    ros_gz_bridge::BridgeConfig config3;
    config3.ros_topic_name = "x500_pose";
    config3.gz_topic_name = "/model/my_x500_0/pose";
    config3.ros_type_name = "geometry_msgs/msg/PoseStamped";
    config3.gz_type_name = "gz.msgs.Pose";
    config3.is_lazy = lazy_subscription;
    
    // camera joint 1
    ros_gz_bridge::BridgeConfig config4;
    config4.ros_topic_name = "rotor1_cmd";
    config4.gz_topic_name = "/rotor1_cmd";
    config4.ros_type_name = "std_msgs/msg/Float64";
    config4.gz_type_name = "gz.msgs.Double";
    config4.is_lazy = lazy_subscription;

    // camera joint 2
    ros_gz_bridge::BridgeConfig config5;
    config5.ros_topic_name = "rotor2_cmd";
    config5.gz_topic_name = "/rotor2_cmd";
    config5.ros_type_name = "std_msgs/msg/Float64";
    config5.gz_type_name = "gz.msgs.Double";
    config5.is_lazy = lazy_subscription;

    // camera joint 3
    ros_gz_bridge::BridgeConfig config6;
    config6.ros_topic_name = "rotor3_cmd";
    config6.gz_topic_name = "/rotor3_cmd";
    config6.ros_type_name = "std_msgs/msg/Float64";
    config6.gz_type_name = "gz.msgs.Double";
    config6.is_lazy = lazy_subscription;

    // 添加配置
    bridge_node->add_bridge(config1);
    bridge_node->add_bridge(config2);
    bridge_node->add_bridge(config3);
    bridge_node->add_bridge(config4);
    bridge_node->add_bridge(config5);
    bridge_node->add_bridge(config6);


    rclcpp::spin(bridge_node);

    // Wait for gz node shutdown
    ignition::transport::waitForShutdown();

    return 0;
}
