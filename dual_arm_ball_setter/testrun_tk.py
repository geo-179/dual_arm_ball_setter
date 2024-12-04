# Import the format for the condition number message
from std_msgs.msg import Float64

import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from dual_arm_ball_setter.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW6 P1.
from hw6code.KinematicChain     import KinematicChain
import random

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain_1 = KinematicChain(node, 'base', 'panda_1_hand', self.jointnames_1())
        self.chain_2 = KinematicChain(node, 'base', 'panda_2_paddle', self.jointnames_2())

        self.q0_1 = np.radians(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.qdot0_1 = np.radians(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


        self.qd_1 = self.q0_1
        self.lam_1 = 20
        self.lam_s_1 = 5
        self.gam_1 = 0.1

        self.q0_2 = np.radians(np.array([0.0, np.deg2rad(46.5675), 0.0, np.deg2rad(-93.1349), 0.0, 0.0, np.deg2rad(46.5675)]))
        self.p0_2 = np.array([0.0, 0.0, 1.5])
        self.R0_2 = Reye()
        

        self.qd_2 = self.q0_2
        self.lam_2 = 20
        self.lam_s_2 = 5
        self.gam_2 = 0.1


        # Drop ball in random location in 0.5 unit radius of base frame zero
        (x,y) = (random.uniform(-0.5, 0.5), random.uniform(-0.2, 0.2))
        

        self.pball = np.array([x, y, 5.0])
        self.vball = np.array([0.0, 0.0, 0.0])
        self.aball = np.array([0.0, 0.0, -5]) # OG -9.81


        # BALL INTERMEDIATE POS TRACKER
        self.pball_init = np.array([x, y, 5.0])
        self.pball_inter = np.array([0,0, 1.5])

        self.paddle_radius = 0.25

    def jointnames(self):
        # Combine joint names from both arms.
        return self.jointnames_1() + self.jointnames_2() + ['panda_1_finger_joint1',
                                      'panda_1_finger_joint2', 
                                      'panda_2_finger_joint1',
                                      'panda_2_finger_joint2']

    # Declare the joint names for the first arm.
    def jointnames_1(self):
        return ['panda_1_joint1',
                'panda_1_joint2',
                'panda_1_joint3', 
                'panda_1_joint4',
                'panda_1_joint5',
                'panda_1_joint6',
                'panda_1_joint7']

    # Declare the joint names for the second arm.
    def jointnames_2(self):
        return ['panda_2_joint1',
                'panda_2_joint2',
                'panda_2_joint3',
                'panda_2_joint4',
                'panda_2_joint5',
                'panda_2_joint6',
                'panda_2_joint7']

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        
        T = np.sqrt(-2*abs(self.p0_2[2] - self.pball_init[2])/self.aball[2])

        if t <= T:
            # Define trajectory/path
            (pd_2, vd_2) = goto(t, T, self.p0_2, np.array([self.pball[0], self.pball[1], 1.5]))
            self.pball_inter[0] = self.pball[0]
            self.pball_inter[1] = self.pball[1]

        else:
            pd_2 = np.array([self.pball_inter[0], self.pball_inter[1], 1.5])
            vd_2 = np.zeros(3)


        Rd_2 = Roty(-np.pi/3)
        wd_2 = np.zeros(3)

        xddot_2 = np.concatenate((vd_2, wd_2))
        
        (ptip_2, Rtip_2, Jv_2, Jw_2) = self.chain_2.fkin(self.qd_2)
        J_2 = np.vstack((Jv_2, Jw_2))

        gamma = 1
        J_weighted_inv_2 = np.linalg.pinv(J_2.T @ J_2 + gamma**2 + np.eye(J_2.shape[1])) @ J_2.T

        ep_2 = pd_2 - ptip_2
        eR_2 = 0.5 * (cross(Rtip_2[0:3,0], Rd_2[0:3,0]) +
                    cross(Rtip_2[0:3,1], Rd_2[0:3,1]) +
                    cross(Rtip_2[0:3,2], Rd_2[0:3,2]))
        error_2 = np.concatenate((ep_2, eR_2))

        qddot_2 = J_weighted_inv_2 @ (xddot_2 + self.lam_2*error_2)
        qd_2 = self.qd_2 + dt*qddot_2

        self.qd_2 = qd_2

        # Return the desired joint and task (position/orientation) pos/vel.
        qd = np.concatenate((self.q0_1, qd_2, [0.0, 0.0, 0.0, 0.0]))
        qddot = np.concatenate((self.qdot0_1, qddot_2, [0.0, 0.0, 0.0, 0.0]))
        
        self.vball += dt * self.aball
        self.pball += dt * self.vball

        # Determine if the ball is within the boundary of the paddle
        r = self.pball - ptip_2
        n = Rtip_2[0:3,0] / np.linalg.norm(Rtip_2[0:3,0])
        if n @ r < 1e-2 and np.linalg.norm(r) <= self.paddle_radius:
            v_normal = np.dot(self.vball, n) * n
            v_plane = self.vball - v_normal
            self.vball = (v_plane - v_normal)

        return (qd, qddot, self.pball)


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('dual_arm_generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()