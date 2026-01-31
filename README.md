1. Project Name: Vision-Based Maze Solving TurtleBot with AI Path
Planning

2. Abstract
Abstract—Autonomous navigation within unstructured and previously unmapped environments remains a foundational challenge in the field of mobile robotics. While classical path-planning
algorithms and end-to-end Deep Reinforcement Learning (DRL) models have demonstrated individual successes, they often sufferfrom rigidity in dynamic settings or instability in continuous
action spaces, respectively. This paper proposes a novel hybrid architectural framework designed for the TurtleBot3 Waffle platform that bifurcates high-level mission logic from low-level
locomotive control. We utilize an OpenCV-based Vision Node for robust environmental state detection via HSV color segmentation,

a Symbolic Planner for deterministic decision-making, and the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm for robust, continuous obstacle avoidance. Experimental 
results conducted within a high-fidelity Gazebo simulation indicate that this hybrid approach significantly mitigates the overestimation bias common in actor-critic methods and provides superior
trajectory smoothness compared to baseline Deep Deterministic Policy Gradient (DDPG) agents. The system achieved a 100% mission termination success rate upon visual goal recognition,
validating the efficacy of symbolic supervision in reinforcement learning tasks.
Index Terms—Deep Reinforcement Learning, TD3, ROS2,Symbolic Reasoning, Computer Vision, Autonomous Navigation, Sim-to-Real.

3. Vision Module

Real-time wall detection using edge detection and Hough transforms
Door/opening identification with contour analysis
Color-based clue recognition (colored markers, symbols)
Visual landmark tracking for SLAM integration

Path Planning

A* algorithm for optimal path finding

AI Reasoning Engine

Symbolic representation of maze elements
Puzzle solving logic (sequence detection, pattern matching)

4. Hardware

TurtleBot3 (Waffle)
Raspberry Pi Camera or Intel RealSense
Ubuntu 20.04 with ROS Noetic

5. Software Dependencies
sudo apt-get install ros-noetic-desktop-full

1.# Python dependenciesKey Components
pip install numpy opencv-python scikit-image matplotlib
pip install pillow scipy networkx

# ROS Package
sudo apt-get install ros-noetic-navigation
sudo apt-get install ros-noetic-vision-opencv
sudo apt-get install ros-noetic-turtlebot3-*

2. Simulation Mode (Gazebo)
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

6. Key Components
   1. Vision Node (vision_node)
   2. Planner Node (planner_node)
   3. Motion Node (notion_node)
   4. AI Node (ai_node)
source install/setup.bash
ros2 run tb3_vision_maze motion_node
