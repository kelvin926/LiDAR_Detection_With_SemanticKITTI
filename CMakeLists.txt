cmake_minimum_required(VERSION 3.5)
project(hyunseo_lidar_detection)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# Install Python scripts
install(PROGRAMS
  scripts/bag_to_pcd.py
  scripts/process_data.py
  DESTINATION lib/${PROJECT_NAME}/scripts
)

# Install Python nodes
install(PROGRAMS
  nodes/human_detection_node.py
  DESTINATION lib/${PROJECT_NAME}/nodes
)

# Install models and other files
install(DIRECTORY launch config models data
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
