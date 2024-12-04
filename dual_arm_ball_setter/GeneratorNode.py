'''GeneratorNode.py

   This creates a trajectory generator node

   To use import

     from GeneratorNode import GeneratorNode

   and call

     generator = GeneratorNode(name, rate, TrajectoryClass)

   This initializes the node, under the specified name and rate.  This
   also requires a trajectory class which must implement:

       trajectory = TrajectoryClass(node)
       jointnames = trajectory.jointnames()
       (desired)  = trajectory.evaluate(t, dt)

   where jointnames is a python list of joint names, which must match
   the URDF (moving) joint names.

   The evaluation is provided the current time (t) and the (dt) some
   the last evaluation, to be used for integration.  It may return

       None                                 Trajectory ends (node shuts down)
       (q, qdot)                            Joint position, velocity
       (q, qdot, p, v)                      Joint and task translation
       (q, qdot, p, v, R, omega)            Joint and task full pose
       (None, None, p, v)                   Just the task translation
       (None, None, None, None, R, omega)   Just the task orientation


   Node:        /generator
   Publish:     /joint_states           sensor_msgs/msg/JointState
                /pose                   geometry_msgs/msg/PoseStamped
                /twist                  geometry_msgs/msg/TwistStamped

'''

import numpy as np
import rclpy
import tf2_ros

from math import nan

from asyncio            import Future
from rclpy.node         import Node
from geometry_msgs.msg  import PoseStamped, TwistStamped
from geometry_msgs.msg  import TransformStamped
from sensor_msgs.msg    import JointState

from hw3code.TransformHelpers   import quat_from_R, Point_from_p

from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Point, Vector3, Quaternion
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

#
#   Trajectory Generator Node Class
#
#   This inherits all the standard ROS node stuff, but adds
#     1) an update() method to be called regularly by an internal timer,
#     2) a spin() method, aware when a trajectory ends,
#     3) a shutdown() method to stop the timer.
#
#   Take the node name, the update frequency, and the trajectory class
#   as arguments.
#
class GeneratorNode(Node):
    # Initialization.
    def __init__(self, name, rate, Trajectory):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Set up a trajectory.
        self.trajectory = Trajectory(self)
        self.jointnames = self.trajectory.jointnames()

        # Add a publisher to send the joint commands.
        self.pubjoint = self.create_publisher(JointState, '/joint_states', 10)
        self.pubpose  = self.create_publisher(PoseStamped, '/pose', 10)
        self.pubtwist = self.create_publisher(TwistStamped, '/twist', 10)

        # Add a publisher to send ball marker position
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.pubball = self.create_publisher(MarkerArray, '/visualization_marker_array', quality)

        # Create the sphere marker.
        self.radius = 0.05
        
        diam        = 2 * self.radius
        self.marker = Marker()
        self.marker.header.frame_id  = "base"
        self.marker.header.stamp     = self.get_clock().now().to_msg()
        self.marker.action           = Marker.ADD
        self.marker.ns               = "point"
        self.marker.id               = 1
        self.marker.type             = Marker.SPHERE
        self.marker.pose.orientation = Quaternion()
        self.marker.scale            = Vector3(x = diam, y = diam, z = diam)
        self.marker.color            = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

        # Create the marker array message.
        self.markerarray = MarkerArray(markers = [self.marker])

        # Initialize a regular and static transform broadcaster
        self.tfbroadcaster = tf2_ros.TransformBroadcaster(self)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        # Create a future object to signal when the trajectory ends,
        # i.e. no longer returns useful data.
        self.future = Future()

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now()+rclpy.time.Duration(seconds=self.dt)

        # Create a timer to keep calculating/sending commands.
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()

    # Spin
    def spin(self):
        # Keep running (taking care of the timer callbacks and message
        # passing), until interrupted or the trajectory is complete
        # (as signaled by the future object).
        rclpy.spin_until_future_complete(self, self.future)

        # Report the reason for shutting down.
        if self.future.done():
            self.get_logger().info("Stopping: " + self.future.result())
        else:
            self.get_logger().info("Stopping: Interrupted")


    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        # Compute the trajectory for this time.
        des = self.trajectory.evaluate(self.t, self.dt)
        if des is None:
            self.future.set_result("Trajectory has ended")
            return

        (q, qdot, pball) = des


        # Check the joint results.
        if q     is None:    q     = [nan] * len(self.jointnames)
        if qdot  is None:    qdot  = [nan] * len(self.jointnames)
        if pball is None:    pball = [0.0, 0.0, 0.0]

        # Turn into lists.
        if type(q).__module__     == np.__name__: q        = q.flatten().tolist()
        if type(qdot).__module__  == np.__name__: qdot     = qdot.flatten().tolist()
        if type(pball).__module__ == np.__name__: pball    = pball.flatten().tolist()

        # Verify the sizes.
        if not (len(q) == len(self.jointnames) and len(qdot) == len(self.jointnames)):
            print(q)
            print(qdot)
            raise ValueError("(q) and (qdot) must be same len as jointnames!")
        if not (len(pball) == 3):
            raise ValueError("(pball) must be length 3!")

        # Determine the corresponding ROS time (seconds since 1970).
        now = self.start + rclpy.time.Duration(seconds=self.t)

        # Build up a joint message and publish.
        msg = JointState()
        msg.header.stamp = now.to_msg()         # Current time for ROS
        msg.name         = self.jointnames      # List of joint names
        msg.position     = q                    # List of joint positions
        msg.velocity     = qdot                 # List of joint velocities
        self.pubjoint.publish(msg)

        # Build up marker message and publish
        self.marker.header.stamp  = self.get_clock().now().to_msg()
        self.marker.pose.position = Point_from_p(pball)
        self.pubball.publish(self.markerarray)
