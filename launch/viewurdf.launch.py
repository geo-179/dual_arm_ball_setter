"""Launch a given URDF in RVIZ

This launch file is intended only to help visualize the various robot
models.  To use, run:

   ros2 launch robots viewurdf.launch.py urdf:=<URDF-FILENAME>

This should start
  1) RVIZ, ready to view the robot
  2) The robot_state_publisher to broadcast the robot model
  3) The GUI to move the joints

If RVIZ complains that "Frame [world] does not exist", change the
'Global Options -> Fixed Frame' to something known (it provides a
pull-down menu).

"""

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
# Generate the Launch Description, known the launch parameters
#
def generate_launch_description_internal(context, *args, **kwargs):

    ######################################################################
    # GET THE LAUNCH PARAMETERS
    urdf = LaunchConfiguration('urdf').perform(context)


    ######################################################################
    # LOCATE FILES
    
    # Define the package.
    package = 'robots'

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir(package), 'rviz/viewurdf.rviz')

    # Check the existance of the URDF file
    if urdf == '':
        print("Please add the URDF file with:  'urdf:=<FILENAME>'")
        return []
    if not os.path.exists(urdf):
        print("URDF file '%s' does not exist" % urdf)
        return []
    print("Using URDF file '%s'" % urdf)

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
    
    # Configure a node for the GUI
    node_gui = Node(
        name       = 'gui', 
        package    = 'joint_state_publisher_gui',
        executable = 'joint_state_publisher_gui',
        output     = 'screen',
        on_exit    = Shutdown())


    ######################################################################
    # RETURN THE ELEMENTS IN ONE LIST
    return [
        # Start the robot_state_publisher, RVIZ, and the GUI.
        node_robot_state_publisher,
        node_rviz,
        node_gui,
    ]


#
# Generate the Launch Description
#
def generate_launch_description():
    return LaunchDescription([
        # Declared arguments can be changed in the command line as
        # 'param:=value'

        # Use an argument 'urdf' to select the URDF file.
        DeclareLaunchArgument('urdf', default_value=''),

        # Generate the launch descriptioin, knowing the above arguments.
        OpaqueFunction(function = generate_launch_description_internal)
    ])
