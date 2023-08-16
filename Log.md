# 2023 暑期摸鱼记录

> 暑期实习，实习...
> 
> 干活，干活，干活！
> 
> Dknt 2023.7

预期目标: 基于 ORBSLAM3 的稠密建图。增加避障？

学习一下 ORBSLAM 源码，更改源码，实现稠密建图

任务:

* 仿真平台: ROS 2 Humble + PX4 + Gazebo Garden

    学习 ROS 2. ROS 与 gz 之间的数据交互。

    PX4 不需要 ROS 到 gz 的通信，我可以用 MAVSDK 直接控制无人机。

* 地面站编程 实现无人机速度控制

    地面站编程使用 MAVSDK, 不依赖于 ROS。需要编写一个**控制类**，能同步获取无人机状态

* ORBSLAM 源码，玄学

* 目标检测与云台控制，理论上来讲，不难。实际上来讲，不加也行。

## 2023.7.7

安装 MAVSDK， 运行旧系统的测试代码 **成功**

## 2023.7.8

开始编写外部控制类，添加解锁、起飞等基本功能，添加键盘控制功能 **完成**

## 2023.7.11

添加 gazebo 与 ros2 的通信，写成C++节点 **完成**


## 2023.7.22

添加仿真启动脚本 **完成**

搭建仿真场景 **完成**

## 2023.7.29

编写录包功能，同时记录轨迹，方便离线运行SLAM，并计算误差。 **完成**

问题：模型 sdf 插件中图像帧率设置为 30 fps, 但转发到 ROS 中实际帧率为 25~26, 深度图更低一点。imu 的帧率也是 30, 实际为 120, 高了很多。


## 2023.8.9

这段时间里心情很差，没怎么写这个。做了如下工作：实现了通过移动设备上 QGC 的虚拟摇杆控制仿真中的无人机，这样感觉更真实。为相机添加云台。编写目标检测服务器。上述模块工作正常。

## 2023.8.16

添加了云台稳定控制，基于YOLOv5的相机目标追踪。已完成预期目标，可以写报告了。


了解一下ROS2中导航包的基本使用

ORBSLAM 基本使用