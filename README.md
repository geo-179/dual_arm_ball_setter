# Dual Arm Volleyball Setting Robotics Project
## TK Lee, Gio Huh, Deon Petrizzo
This project involves using two **Emika Franka** robotic arms and a paddle (14 DOF) to imitate a volleyball setting action. The goal is to achieve a reliable and dynamic setting motion, similar to that in volleyball, using **ROS 2 (Humble)**. That is, for any given ball trajectory, it must be able to hit the ball at an angle that results in purely vertical motion, as seen in the following gif.

![Demo Gif](./assets/gif/dual-arm-ball-setter-gif.gif)

We have implemented a **task hierarchy** to ensure that the robot can prioritize and execute tasks effectively, enabling reliable performance in volleyball setting actions. The project utilizes trajectory utilities imported from other projects, though they can be replaced or updated when pulling the repository to run the system.

## Video Demo
https://user-images.githubusercontent.com/geo-179/dual_arm_ball_setter/assets/videos/dual-arm-ball-setter-demo.mp4


You can watch our full demonstration video showcasing the dual-arm volleyball setting action, including different scenarios and successful implementations of the task hierarchy system.

## Technical Documentation
For detailed information about the system architecture, implementation details, and experimental results, please refer to our technical report:

[Read Full Technical Report](./docs/dual-arm-ball-setter-report.pdf)

The technical report includes:
- Detailed system architecture
- Task hierarchy implementation
- Experimental results and analysis
- Future improvements and recommendations

## Getting Started
To run the project, follow these steps:
 - Clone the project repository to your workspace and build the project in ROS 2.
 - Run the following command:
```bash
ros2 launch dual_arm_ball_setter final.launch.py
