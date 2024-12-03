# Import the format for the condition number message
from std_msgs.msg import Float64

import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW6 P1.
from hw6code.KinematicChain     import KinematicChain


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain_1 = KinematicChain(node, 'base', 'panda_1_hand', self.jointnames())
        self.chain_2 = KinematicChain(node, 'base', 'panda_2_paddle', self.jointnames())
        

        self.q0 = np.radians(np.array([0.0, np.deg2rad(46.5675), 0.0, np.deg2rad(-93.1349), 0.0, 0.0, np.deg2rad(46.5675)]))
        self.p0 = np.array([0.0, 0.7, 0.6])
        self.R0 = Reye()

        self.pleft = np.array([0.3, 0.5, 0.15])
        self.pright = np.array([-0.3, 0.5, 0.15])
        self.Rright = Reye()
        self.Rleft = Rotz(np.pi/2) @ Rotx(-np.pi/2)

        self.qd = self.q0
        self.lam = 20
        self.lam_s = 5
        self.gam = 0.1

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['panda_1_joint1',
                'panda_1_joint2',
                'panda_1_joint3',
                'panda_1_joint4',
                'panda_1_joint5',
                'panda_1_joint6',
                'panda_1_joint7',
                'panda_1_joint8',
                'panda_2_joint1',
                'panda_2_joint2',
                'panda_2_joint3',
                'panda_2_joint4',
                'panda_2_joint5',
                'panda_2_joint6',
                'panda_2_joint7',
                'panda_2_joint8']


    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        if t > 14: 
            return None

        # Define trajectory/path
        pd = np.array([0, 0.95 - 0.25*np.cos(t), 0.6 + 0.25*np.sin(t)])
        vd = np.array([0, 0.25*np.sin(t), 0.25*np.cos(t)])
        Rd = self.R0
        wd = np.zeros(3)

        xddot = np.concatenate((vd, wd))
        
        (ptip, Rtip, Jv, Jw) = self.chain_2.fkin(self.qd)
        J = np.vstack((Jv, Jw))

        ep = pd - ptip
        eR = 0.5 * (cross(Rtip[0:3,0], Rd[0:3,0]) +
                    cross(Rtip[0:3,1], Rd[0:3,1]) +
                    cross(Rtip[0:3,2], Rd[0:3,2]))
        error = np.concatenate((ep, eR))

        qddot = np.linalg.pinv(J) @ (xddot + self.lam*error)
        qd = self.qd + dt*qddot

        self.qd = qd

        # Return the desired joint and task (position/orientation) pos/vel.
        return (qd, qddot, pd, vd, Rd, wd)


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()