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
        self.chain_to_hand_2 = KinematicChain(node, 'base', 'panda_2_hand', self.jointnames_2())

        self.q0_1 = np.radians(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.qdot0_1 = np.radians(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.qd_1 = self.q0_1

        self.q0_2 = np.radians(np.array([0.0, np.deg2rad(46.5675), 0.0, np.deg2rad(-93.1349), 0.0, 0.0, np.deg2rad(46.5675)]))
        self.p0_2 = np.array([0.0, 0.0, 1.5])
        self.R0_2 = Reye()
        self.qd_2 = self.q0_2
        
        self.lam = 20
        self.gam = 0.1

        # Drop ball in random location in 0.1 unit radius of base frame zero
        (x,y) = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))

        self.pball = np.array([x, y, 5.0])
        self.vball = np.array([0.0, 0.0, 0.0])
        self.aball = np.array([0.0, 0.0, -1.0])

        # Intermediate ball position tracking variables
        self.pball_init = np.array([x, y, 5.0])
        self.pball_inter = np.array([0.0, 0.0, 1.5])

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
            (pd_2, vd_2) = goto(t, T, self.p0_2, np.array([self.pball[0], self.pball[1], self.p0_2[2]]))
            self.pball_inter[0] = self.pball[0]
            self.pball_inter[1] = self.pball[1]
        else:
            pd_2 = np.array([self.pball_inter[0], self.pball_inter[1], self.p0_2[2]])
            vd_2 = np.zeros(3)

        #### Chain 2 kinematics ###
        nd_2 = np.array([0, 0, 1])
        wd_2 = np.zeros(3)

        (qd_2, qddot_2, ptip_2, Rtip_2, n) = self.ikin2(self.chain_2, dt, self.qd_2, pd_2, vd_2, nd_2, wd_2)    
        self.qd_2 = qd_2

        #### Chain 1 kinematics ###
        (phand_2, _, _, _) = self.chain_to_hand_2.fkin(self.qd_2)
        hand_2_to_hand_1 = ptip_2 - phand_2

        pd_1 = ptip_2 + hand_2_to_hand_1
        vd_1 = vd_2
        Rd_1 = np.array([Rtip_2[:, 0], -Rtip_2[:, 1], -Rtip_2[:, 2]])
        wd_1 = np.zeros(3)

        (qd_1, qddot_1, _, _) = self.ikin1(self.chain_1, dt, self.qd_1, pd_1, vd_1, Rd_1, wd_1)
        self.qd_1 = qd_1

        # Return the desired joint and task (position/orientation) pos/vel.
        qd = np.concatenate((qd_1, qd_2, np.zeros(4)))
        qddot = np.concatenate((qddot_1, qddot_2, np.zeros(4)))
        
        self.vball += dt * self.aball
        self.pball += dt * self.vball

        # Determine if the ball is in collision with the paddle
        r = self.pball - ptip_2
        if n @ r < 1e-2 and np.linalg.norm(r) <= self.paddle_radius:
            v_normal = (self.vball @ n) * n
            v_plane = self.vball - v_normal
            self.vball = v_plane - v_normal

        return (qd, qddot, self.pball)


    def ikin1(self, chain, dt, qd_last, pd, vd, Rd, wd):
        xddot = np.concatenate((vd, wd))
        
        (ptip, Rtip, Jv, Jw) = chain.fkin(qd_last)

        J = np.vstack((Jv, Jw))

        ep_ = ep(pd, ptip)
        eR_ = eR(Rd, Rtip)
        error = np.concatenate((ep_, eR_))

        qddot = np.linalg.pinv(J.T @ J + self.gam**2 * np.eye(7)) @ J.T @ (xddot + self.lam*error)
        qd = qd_last + dt*qddot

        return (qd, qddot, ptip, Rtip)


    def ikin2(self, chain, dt, qd_last, pd, vd, nd, wd):
        (ptip, Rtip, Jv, Jw) = chain.fkin(qd_last)

        # Create partial orientation jacobian, excluding orientation about axis normal to paddle (x-axis of tip frame)
        A = np.array([[0, 1, 0], [0, 0, 1]])
        J = np.vstack((Jv, A @ Rtip.T @ Jw))

        # Similarly, modify wd to keep the equation wd = Jw*qdot consistent
        xddot = np.concatenate((vd, A @ Rtip.T @ wd))

        # Get unit normal vector to paddle surface (x-basis vector)
        n = Rtip[:,0]

        ep_ = ep(pd, ptip)
        en = np.cross(n, nd)
        error = np.concatenate((ep_, en))

        damping = self.gam**2 * np.eye(7)
        qddot = np.linalg.pinv(J.T @ J + damping) @ J.T @ (xddot + self.lam * error)

        qd = qd_last + dt*qddot

        return (qd, qddot, ptip, Rtip, n)


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('dual_arm_generator', 500, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()