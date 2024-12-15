# Dual Arm Volleyball Setting Robotics Project
## TK Lee, Gio Huh, Deon Petrizzo
This project involves using two **Emika Franka** robotic arms and a paddle to imitate a volleyball setting action. The goal is to achieve a reliable and dynamic setting motion, similar to that in volleyball, using **ROS 2 (Humble)**.

We have implemented a **task hierarchy** to ensure that the robot can prioritize and execute tasks effectively, enabling reliable performance in volleyball setting actions. The project utilizes trajectory utilities imported from other projects, though they can be replaced or updated when pulling the repository to run the system.

## Getting Started

To run the project, follow these steps:

Clone the project repository to your workspace and build the project in ros2. Then, run the following command:
```bash
ros2 launch dual_arm_ball_setter final.launch.py 