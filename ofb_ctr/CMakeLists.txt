cmake_minimum_required(VERSION 3.8)
project(ofb_ctr)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

##########################################################################
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED) # 自定义话题、服务生成
find_package(cv_bridge REQUIRED)
find_package(gz-transport12 REQUIRED)
find_package(gz-msgs9 REQUIRED)
find_package(Eigen3 REQUIRED)

set(GZ_TRANSPORT_VER ${gz-transport12_VERSION_MAJOR})
set(GZ_MSGS_VER ${gz-msgs9_VERSION_MAJOR})
set(GZ_TARGET_PREFIX gz)

message(STATUS "Compiling against Gazebo Garden")

find_package(MAVSDK REQUIRED)

include_directories(
  "/usr/include/mavsdk"
  "include/ofb_ctr"
  "/usr/include/eigen3"
)

# 生成服务库
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/TargetDetection.srv"
  DEPENDENCIES sensor_msgs std_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)


# 外部控制库   这里编译为静态库，否则会有链接错误
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(OffboardCtr STATIC src/OffboardCtr.cpp)
target_link_libraries(OffboardCtr
  MAVSDK::mavsdk
  ${OpenCV_LIBS}
  ${cpp_typesupport_target}
)
ament_target_dependencies(OffboardCtr
  rclcpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
)


# 测试代码
add_executable(my_test src/my_test.cpp)
target_link_libraries(my_test OffboardCtr)
ament_target_dependencies(my_test rclcpp)

# gz 消息转发
add_executable(gz_ros_bridge src/gz_ros_bridge.cpp)
target_link_libraries(gz_ros_bridge
  ${GZ_TARGET_PREFIX}-msgs${GZ_MSGS_VER}::core
  ${GZ_TARGET_PREFIX}-transport${GZ_TRANSPORT_VER}::core
)
ament_target_dependencies(gz_ros_bridge
  "image_transport"
  "ros_gz_bridge"
  "rclcpp"
  "sensor_msgs"
)

# 目标检测服务测试
add_executable(detect_client_test src/detect_client_test.cpp)
target_link_libraries(detect_client_test
  ${OpenCV_LIBS}
  ${cpp_typesupport_target}
)
ament_target_dependencies(detect_client_test
  rclcpp
  sensor_msgs
  cv_bridge
)


# 添加到 install 目录
install(TARGETS
  my_test
  gz_ros_bridge
  detect_client_test
  DESTINATION lib/${PROJECT_NAME}
)
##########################################################################




if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
