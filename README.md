1. Project Name: Vision-Based Maze Solving TurtleBot with AI Path Planning

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
Repository Link: https://github.com/Alasadi85/tb3-vision-maze-project 

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
# AI-First Maze Solver (TurtleBot3 DRL)

This project implements a Deep Reinforcement Learning (TD3) agent for autonomous maze navigation using ROS 2 Humble and Gazebo.

## Installation & Setup

### 1. Prerequisites
*   Ubuntu 22.04 LTS
*   **ROS 2 Humble** (Desktop Full)
*   **Gazebo 11**
*   **Python Libraries:**
    ```bash
    pip install torch numpy
    ```

### 2. Build the Workspace
1.  Open a terminal in this folder (where `src` and `models` are located).
2.  Build the ROS 2 package:
    ```bash
    colcon build --symlink-install
    ```
3.  Source the overlay:
    ```bash
    source install/setup.bash
    ```

## Usage

### Run the AI Demo (Pre-trained)
To see the robot solve the maze using the loaded models:
```bash
ros2 launch tb3_vision_maze deep_learning_demo.launch.py
```
*Note: Ensure you are in the root of this workspace so the code can find the `./models` directory.*

### Run Training
To train the agent from scratch (or continue training):
```bash
ros2 launch tb3_vision_maze train_maze.launch.py
```

---

# AI Implementation Details: Deep Reinforcement Learning (TD3)
## Master's Project Specialist Report

This document details the **Artificial Intelligence** subsystem of the project. It provides the theoretical justification, mathematical formulation, and practical implementation details of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm used for autonomous navigation.

---

## 1. Why Deep Reinforcement Learning (DRL)?
Traditional path planning (like A* or Dijkstra) requires a known map (SLAM). Since our objective is to navigate **unknown** environments using only local sensor data, we frame this as a **Continuous Control Problem**.

We use **Model-Free, Off-Policy RL** because:
*   **Model-Free:** The robot does not need to know the physics of the world (friction, maze layout) beforehand; it learns them by trial and error.
*   **Off-Policy:** We can learn from the data collected by a "Teacher" (Expert/Heuristic) to bootstrap learning, rather than only learning from the current policy's mistakes.
*   **Continuous Control:** A differential drive robot (TurtleBot3) moves most smoothly when given continuous velocity commands ($v, \omega$), rather than discrete commands (Forward/Stop/Turn).

---

## 2. The Algorithm: TD3 (Twin Delayed DDPG)
**file:** `td3_agent.py`

We chose **TD3** over DDPG or PPO because DDPG suffers from **Q-Value Overestimation** (it thinks actions are better than they are), which leads to suboptimal policies. TD3 fixes this with three key tricks:

### Trick 1: Clipped Double Q-Learning (Twin Critics)
Instead of one "Critic" network estimating the value of an action, we use **Two** ($Q_1$ and $Q_2$).
When calculating the target value for the bellman update, we take the **minimum** of the two predictions.
$$ y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \pi_{\theta'}(s') + \epsilon) $$
*   **Intuition:** If one critic overestimates, the min() operation ignores it. This prevents the "optimism bias" from exploding.

### Trick 2: Delayed Policy Updates
The Actor (Policy) is updated less frequently than the Critics (Values).
*   **Code:** `if self.total_it % policy_freq == 0:` (Frequency = 2)
*   **Intuition:** If the Critic is unstable/changing rapidly, updating the Actor based on it causes the policy to "chase" a moving target. Waiting for the Critic to settle (converge) leads to more stable policy learning.

### Trick 3: Target Policy Smoothing
We add noise to the *target* action when updating the Critic.
$$ \tilde{a} = \pi_{\theta'}(s') + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c) $$
*   **Hyperparameters:** Noise $\sigma=0.2$, Clip $c=0.5$.
*   **Intuition:** This acts as a regularizer. The value of an action shouldn't change drastically if we slightly perturb the action. "Sharp peaks" in the value function are smoothed out.

---

## 3. Network Architecture (PyTorch)
We typically use Multi-Layer Perceptrons (MLP) for vector inputs (Lidar).

### The Actor Network ($\pi_\theta$)
Responsible for mapping **State $\to$ Action**.
*   **Input Layer:** 28 neurons (State Dim).
*   **Hidden Layer 1:** 400 neurons, ReLU Activation.
*   **Hidden Layer 2:** 300 neurons, ReLU Activation.
*   **Output Layer:** 2 neurons ($v, \omega$), **Tanh Activation**.
    *   **Reason for Tanh:** Outputs values in range $[-1, 1]$. We scale this by `max_action` to translate to physical motor limits.

### The Critic Network ($Q_\phi$)
Responsible for mapping **(State, Action) $\to$ Q-Value** (Expected Return).
*   **Input Layer:** 30 neurons (28 State + 2 Action).
*   **Hidden Layer 1:** 400 neurons, ReLU.
*   **Hidden Layer 2:** 300 neurons, ReLU.
*   **Output Layer:** 1 neuron (Linear, no activation). Represents the predicted future reward sum.

---

## 4. Hyperparameters Table
These are the exact values used in `train_node.py`, critical for reproducibility.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Batch Size** | 100 | Number of experience tuples sampled from Replay Buffer. |
| **Discount Factor ($\gamma$)** | 0.99 | Importance of future rewards (High = long-term planning). |
| **Tau ($\tau$)** | 0.005 | Soft Update rate. Target networks move 0.5% towards online networks per step. |
| **Policy Noise** | 0.2 | Noise added to target actions (Smoothing). |
| **Noise Clip** | 0.5 | Max range of smoothing noise. |
| **Exploration Noise** | 0.1 | Gaussian noise added to actions during data collection (Exploration). |
| **Actor LR** | 3e-4 | Learning Rate for Adam Optimizer. |
| **Critic LR** | 3e-4 | Learning Rate for Adam Optimizer. |

---

## 5. Training Methodology: "Teacher-Student" (DAGGER-inspired)
A pure RL agent starting from random weights will crash thousands of times before finding the exit, wasting hours of training.

**Our Approach:**
1.  **Phase 1: Teacher (Steps 0 - 100k)**
    *   **Controller:** We implemented a `heuristic` logic (Lane Centering + Gap Finding).
    *   **Data Collection:** The robot drives well using this teacher. We record `(state, action, reward, next_state)` into the **Replay Buffer**.
    *   **Result:** The buffer generates a "dataset of success".

2.  **Phase 2: Student (Steps 100k - 1M)**
    *   **Controller:** Switch to the `TD3 Policy`.
    *   **Training:** The Neural Network samples from the buffer. It learns: *"In this situation (State), the Teacher did X (Action), which resulted in high Reward."*
    *   **Self-Improvement:** Eventually, the Student surpasses the Teacher because the Teacher is rigid (can't handle edge cases), whereas the Student continues to optimize the Reward Function given new data.

---

## 6. Mathematical Update Rules (For Defense Slides)

**Critic Loss (Mean Squared Error):**
$$ L(\phi) = \frac{1}{N} \sum (y - Q_{\phi}(s, a))^2 $$
Where target $y$ is the Bellman target (min of twins).

**Actor Loss (Deterministic Policy Gradient):**
$$ \nabla_\theta J(\theta) = \frac{1}{N} \sum \nabla_a Q_{\phi_1}(s, a)|_{a=\pi_\theta(s)} \nabla_\theta \pi_\theta(s) $$
*   **Translation:** "Change the Actor weights $\theta$ in the direction that increases the Q-Value estimated by Critic 1."
