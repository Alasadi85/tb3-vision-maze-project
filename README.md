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

# 🚀 TurtleBot3 Maze Navigation - Complete Setup Guide

## 📋 Prerequisites
- Ubuntu 22.04
- ROS 2 Humble
- TurtleBot3 packages
- Gazebo

## 🖥️ Terminal Setup (11 Terminals Required)

| Terminal | Purpose | Command to Copy |
|----------|---------|-----------------|
| **Terminal 0** | Cleanup | `pkill gzserver`<br>`pkill gzclient` |
| **Terminal 1** | Gazebo Server | ```bash<br>source /opt/ros/humble/setup.bash<br>export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib<br>export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models<br><br>gzserver ~/tb3_project_ws/worlds/maze_world.world \<br>  -s libgazebo_ros_init.so \<br>  -s libgazebo_ros_factory.so<br>``` |
| **Terminal 2** | Gazebo GUI | ```bash<br>source /opt/ros/humble/setup.bash<br>gzclient<br>``` |
| **Terminal 3** | ROS Service Check | ```bash<br>source /opt/ros/humble/setup.bash<br>ros2 service list | grep spawn<br>``` |
| **Terminal 4** | Spawn Robot | ```bash<br>source /opt/ros/humble/setup.bash<br>export TURTLEBOT3_MODEL=waffle<br>ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py<br>``` |
| **Terminal 5** | Camera Check | ```bash<br>source /opt/ros/humble/setup.bash<br>ros2 topic list | grep image<br>``` |
| **Terminal 6** | Vision Node | ```bash<br>source /opt/ros/humble/setup.bash<br>cd ~/tb3_project_ws<br>source install/setup.bash<br>ros2 run tb3_vision_maze vision_node<br>``` |
| **Terminal 7** | Vision Monitor | ```bash<br>source /opt/ros/humble/setup.bash<br>ros2 topic echo /maze_state<br>``` |
| **Terminal 8** | Planner Node | ```bash<br>source /opt/ros/humble/setup.bash<br>cd ~/tb3_project_ws<br>source install/setup.bash<br>ros2 run tb3_vision_maze planner_node<br>``` |
| **Terminal 9** | Planner Monitor | ```bash<br>source /opt/ros/humble/setup.bash<br>ros2 topic echo /planner_cmd<br>``` |
| **Terminal 10** | Motion Node | ```bash<br>source /opt/ros/humble/setup.bash<br>cd ~/tb3_project_ws<br>source install/setup.bash<br>ros2 run tb3_vision_maze motion_node<br>``` |

## 🔄 Execution Order
1. **Terminal 0** → Cleanup
2. **Terminal 1** → Gazebo server
3. **Terminal 2** → Gazebo GUI
4. **Terminal 3** → Verify ROS services
5. **Terminal 4** → Spawn robot
6. **Terminal 5** → Check camera topics
7. **Terminal 6** → Start vision node
8. **Terminal 7** → Monitor vision output
9. **Terminal 8** → Start planner node
10. **Terminal 9** → Monitor planner commands
11. **Terminal 10** → Start motion node

## 📊 System Architecture
ros2 run tb3_vision_maze motion_node

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
