cmake_minimum_required(VERSION 3.8)
project(bag_record)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# ? 为什么需要 gz
find_package(ros_gz_bridge REQUIRED)
find_package(cv_bridge REQUIRED)


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

# SLAM 数据集录制
add_executable(sensor_recorder src/sensor_recorder.cpp)
ament_target_dependencies(sensor_recorder rclcpp sensor_msgs cv_bridge)
target_link_libraries(sensor_recorder ${OpenCV_LIBS})


install(TARGETS
  sensor_recorder
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
