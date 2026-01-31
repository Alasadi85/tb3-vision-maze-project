Terminal 0
pkill gzserver
pkill gzclient

Terminal 1 (Gazebo Server Maze + ROS bridge)
source /opt/ros/humble/setup.bash
export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib
export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models

gzserver ~/tb3_project_ws/worlds/maze_world.world \
  -s libgazebo_ros_init.so \
  -s libgazebo_ros_factory.so

Terminal 2 (Gazebo GUI)
source /opt/ros/humble/setup.bash
gzclient

Terminal 3 (Gazebo bridge check + ROS)
source /opt/ros/humble/setup.bash
ros2 service list | grep spawn

Terminal 4 (Spawn TurtleBot3)
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py

Terminal 5 (Camera topic check)
source /opt/ros/humble/setup.bash
ros2 topic list | grep image

Terminal 6 (Vision node)
source /opt/ros/humble/setup.bash
cd ~/tb3_project_ws
source install/setup.bash
ros2 run tb3_vision_maze vision_node

Terminal 7 (Vision output check)
source /opt/ros/humble/setup.bash
ros2 topic echo /maze_state

Terminal 8 (Planner node)
source /opt/ros/humble/setup.bash
cd ~/tb3_project_ws
source install/setup.bash
ros2 run tb3_vision_maze planner_node

Terminal 9 (Planner output check)
source /opt/ros/humble/setup.bash
ros2 topic echo /planner_cmd

Terminal 10 (Motion node)
source /opt/ros/humble/setup.bash
cd ~/tb3_project_ws
source install/setup.bash
ros2 run tb3_vision_maze motion_node
