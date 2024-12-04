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

from hw3code.TransformHelpers   import quat_from_R


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

        # Extract the appropriate information.
        if   len(des) == 2:
            (q,qdot,p,v,R,w) = (des[0],des[1],None,None,None,None)
        elif len(des) == 4:
            (q,qdot,p,v,R,w) = (des[0],des[1],des[2],des[3],None,None)
        elif len(des) == 6:
            (q,qdot,p,v,R,w) = des
        else:
            raise ValueError("The trajectory must return 2, 4, 6 elements")

        # Check the joint results.
        if q    is None:    q    = [nan] * len(self.jointnames)
        if qdot is None:    qdot = [nan] * len(self.jointnames)
        if p    is None:    p    = [0.0, 0.0, 0.0]
        if v    is None:    v    = [0.0, 0.0, 0.0]
        if w    is None:    w    = [0.0, 0.0, 0.0]

        if R    is None:    quat  = [0.0, 0.0, 0.0, 1.0]
        else:               quat  = quat_from_R(R)

        # Turn into lists.
        if type(q).__module__    == np.__name__: q    = q.flatten().tolist()
        if type(qdot).__module__ == np.__name__: qdot = qdot.flatten().tolist()
        if type(p).__module__    == np.__name__: p    = p.flatten().tolist()
        if type(v).__module__    == np.__name__: v    = v.flatten().tolist()
        if type(w).__module__    == np.__name__: w    = w.flatten().tolist()

        # Verify the sizes.
        if not (len(q) == len(self.jointnames) and
                len(qdot) == len(self.jointnames)):
            print(q)
            print(qdot)
            raise ValueError("(q) and (qdot) must be same len as jointnames!")
        if not (len(p) == 3 and len(v) == 3):
            raise ValueError("(p) and (v) must be length 3!")
        if not (len(w) == 3):
            raise ValueError("(omega) must be length 3!")

        # Determine the corresponding ROS time (seconds since 1970).
        now = self.start + rclpy.time.Duration(seconds=self.t)

        # Build up a joint message and publish.
        msg = JointState()
        msg.header.stamp = now.to_msg()         # Current time for ROS
        msg.name         = self.jointnames      # List of joint names
        msg.position     = q                    # List of joint positions
        msg.velocity     = qdot                 # List of joint velocities
        self.pubjoint.publish(msg)

        # Build up a pose message and publish.
        msg = PoseStamped()
        msg.header.stamp       = now.to_msg()   # Current time for ROS
        msg.pose.position.x    = p[0]
        msg.pose.position.y    = p[1]
        msg.pose.position.z    = p[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.pubpose.publish(msg)

        # Build up a twist message and publish.
        msg = TwistStamped()
        msg.header.stamp    = now.to_msg()      # Current time for ROS
        msg.twist.linear.x  = v[0]
        msg.twist.linear.y  = v[1]
        msg.twist.linear.z  = v[2]
        msg.twist.angular.x = w[0]
        msg.twist.angular.y = w[1]
        msg.twist.angular.z = w[2]
        self.pubtwist.publish(msg)

        # Prepare a transform message and broadcast.
        msg = TransformStamped()
        msg.header.stamp            = now.to_msg()
        msg.header.frame_id         = 'world'
        msg.child_frame_id          = 'desired'
        msg.transform.translation.x = p[0]
        msg.transform.translation.y = p[1]
        msg.transform.translation.z = p[2]
        msg.transform.rotation.x    = quat[0]
        msg.transform.rotation.y    = quat[1]
        msg.transform.rotation.z    = quat[2]
        msg.transform.rotation.w    = quat[3]
        self.tfbroadcaster.sendTransform(msg)
