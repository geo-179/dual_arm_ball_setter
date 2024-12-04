import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import DeclareLaunchArgument
from launch.actions                    import OpaqueFunction
from launch.actions                    import Shutdown
from launch.substitutions              import LaunchConfiguration
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('dual_arm_ball_setter'), 'rviz/viewurdf.rviz')

    # Locate the URDF file.
    urdf = os.path.join(pkgdir('dual_arm_ball_setter'), 'urdf/dual_panda_example.urdf')

    # Load the robot's URDF file (XML).
    with open(urdf, 'r') as file:
        robot_description = file.read()

    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the robot_state_publisher.
    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    # Configure a node for RVIZ
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())

    # Configure a node for the joint trajectory
    node_trajectory = Node(
        name       = 'trajectory', 
        package    = 'dual_arm_ball_setter',
        executable = 'testrun_2',
        output     = 'screen')

    ######################################################################
    # RETURN THE ELEMENTS IN ONE LIST
    return LaunchDescription([
        # Start the robot_state_publisher, RVIZ, and the trajectory.
        node_robot_state_publisher,
        node_rviz,
        node_trajectory,
    ])