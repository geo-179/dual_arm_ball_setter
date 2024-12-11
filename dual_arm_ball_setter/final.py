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
        self.N_ARM = 7             # Number of joints per arm
        self.N_COMBINED = 14       # Total number of joints of combined system
        self.PADDLE_RADIUS = 0.25  

        ##################### ROBOT INITIALIZATION #####################
        # Set up the kinematic chain objects
        self.chain_1 = KinematicChain(node, 'base', 'panda_1_hand', self.jointnames_1())
        self.chain_2 = KinematicChain(node, 'base', 'panda_2_paddle', self.jointnames_2())
        self.chain_to_hand_2 = KinematicChain(node, 'base', 'panda_2_hand', self.jointnames_2())

        self.Jblank = np.zeros((3, self.N_COMBINED))
        
        indexlist = lambda start, n: list(range(start, start+n))
        self.i_1 = indexlist(0, self.N_ARM)
        self.i_2 = indexlist(self.N_ARM, self.N_ARM)

        self.qd_1 = np.radians(np.zeros(self.N_ARM))
        self.qddot_1 = np.radians(np.zeros(self.N_ARM))
        self.qd_2 = np.radians(np.zeros(self.N_ARM))
        self.qddot_2 = np.radians(np.zeros(self.N_ARM))

        self.p0_2 = np.array([0.0, 0.0, 1.5])

        self.lam = 20
        self.gam = 0.1

        # Desired position and orientaion of chain1 tip in coordinates of and relative to chain2 tip
        self.pd_12 = np.array([0.0, 0.0, self.PADDLE_RADIUS])  
        self.Rd_12 = Rotx(np.pi)


        ##################### BALL INITIALIZATION #####################
        g = 1

        p0_ball = np.array([random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(0.0, 10.0)])
        p1_ball = np.array([random.uniform(-0.5, 0.5), random.uniform(-1.0, 1.0), random.uniform(1.0, 2.0)])

        # determined v0 that satisfies p0 and p1 start and end positions
        v0_ball = np.zeros(3)
        pr_ball = p0_ball + p1_ball

        z = np.array([0.0, 0.0, 1.0])
        pr_proj = np.array([pr_ball[0], pr_ball[1], 0.0])
        get_theta = lambda v1, v2: np.arccos((v1 @ v2) / np.norm(v1 @ v2))
        th_r0 = get_theta(v0_ball, pr_ball)
        th_r1 = get_theta(v1_ball, pr_ball)
        th_r0_max = get_theta(pr_ball, z)
        th_r1_max = get_theta(pr_ball, pr_proj)

        self.pball = p0_ball
        self.vball = v0_ball
        self.aball = np.array([0.0, 0.0, -g])

        # Intermediate ball position tracking variables
        self.pball_init = np.array([x, y, 5.0])
        self.pball_inter = np.array([0.0, 0.0, 1.5])
        self.vball_last = self.vball
        self.aball_last = self.aball

    def jointnames(self):
        ''' Combined joint names for both arms '''
        return self.jointnames_1() + self.jointnames_2() + ['panda_1_finger_joint1',
                                      'panda_1_finger_joint2', 
                                      'panda_2_finger_joint1',
                                      'panda_2_finger_joint2']


    def jointnames_1(self):
        ''' Declare the joint names for the first arm '''
        return ['panda_1_joint1',
                'panda_1_joint2',
                'panda_1_joint3', 
                'panda_1_joint4',
                'panda_1_joint5',
                'panda_1_joint6',
                'panda_1_joint7']


    def jointnames_2(self):
        ''' Declare the joint names for the second arm '''
        return ['panda_2_joint1',
                'panda_2_joint2',
                'panda_2_joint3',
                'panda_2_joint4',
                'panda_2_joint5',
                'panda_2_joint6',
                'panda_2_joint7']


    def evaluate(self, t, dt):
        ''' Evaluate at the given time. This was last called (dt) ago '''
        T = np.sqrt(-2*abs(self.p0_2[2] - self.pball_init[2])/self.aball[2])
        if t <= T:
            # (pd_2, vd_2) = goto(t, T, self.p0_2, np.array([self.pball[0], self.pball[1], self.p0_2[2]]))
            self.pball_inter[0] = self.pball[0]
            self.pball_inter[1] = self.pball[1]
            nd = -self.vball / np.norm(self.vball)
            nddot = -self.aball / np.norm(self.aball)
            wd = np.cross(nd, nddot)
            vball_last = self.vball
            aball_last = self.aball
        else:
            # pd_2 = np.array([self.pball_inter[0], self.pball_inter[1], self.p0_2[2]])
            # vd_2 = np.zeros(3)
            nd = -self.vball_last / np.norm(self.vball_last)
            nddot = -self.aball_last
    
        (qd, qddot, ptip_2, n) = self.ikin(dt, wd, nd)
        self.qd_1 = qd[self.i_1]
        self.qd_2 = qd[self.i_2]

        # Return the desired joint and task (position/orientation) pos/vel.
        qd = np.concatenate((qd, np.zeros(4)))
        qddot = np.concatenate((qddot, np.zeros(4)))
        
        self.vball += dt * self.aball
        self.pball += dt * self.vball

        # Determine if the ball is in collision with the paddle
        r = self.pball - ptip_2
        if n @ r < 1e-2 and np.linalg.norm(r) <= self.PADDLE_RADIUS:
            v_normal = (self.vball @ n) * n
            v_plane = self.vball - v_normal
            self.vball = v_plane - v_normal

        return (qd, qddot, self.pball)


    def ikin(self, dt, wd, nd):
        ''' Compute the inverse kinematics of the combined 14 DOF system
            to achieve the desired primary, secondary, tertiary, and quaternary tasks '''

        # Calculate the individual chains
        (ptip_1, Rtip_1, Jvbar_1, Jwbar_1) = self.chain_1.fkin(self.qd_1)
        (ptip_2, Rtip_2, Jvbar_2, Jwbar_2) = self.chain_2.fkin(self.qd_2)

        # Expand the jacobians
        Jv_1 = self.Jblank.copy()  
        Jw_1 = self.Jblank.copy() 
        Jv_2 = self.Jblank.copy()  
        Jw_2 = self.Jblank.copy() 

        Jv_1[:, self.i_1] = Jvbar_1
        Jw_1[:, self.i_1] = Jwbar_1
        Jv_2[:, self.i_2] = Jvbar_2
        Jw_2[:, self.i_2] = Jwbar_2

        # Combine chains 1 and 2 into 12 (arm 1 hand w.r.t arm 2 paddle center)
        p_12 = Rtip_2.T @ (ptip_1 - ptip_2)
        R_12 = Rtip_2.T @ Rtip_1
        Jv_12 = Rtip_2.T @ (Jv_1 - Jv_2 + crossmat(ptip_1 - ptip_2) @ Jw_2)
        Jw_12 = Rtip_2.T @ (Jw_1 - Jw_2)

        # Calculate joint velocities for the primary task -- keeping hands together
        J_p = np.vstack((Jv_12, Jw_12))
        ep_12 = ep(self.pd_12, p_12)
        eR_12 = eR(self.Rd_12, R_12)
        error_p = np.concatenate((ep_12, eR_12))
        xddot_p = np.zeros(6)
        primary = self.weighted_inv(J_p) @ (xddot_p + self.lam*error_p)

        # Calculate joint velocities for the secondary task -- normal
        A = np.array([[0, 1, 0], [0, 0, 1]])
        J_s = self.nullspace(J_p)
        J_s = A @ Rtip_2.T @ Jw_2

        n = Rtip_2[:,0]  # Get unit normal vector to paddle surface (x-basis vector)
        en = np.cross(n, nd)
        wr = (wd + self.lam * en)
        xrdot_s = A @ Rtip_2.T @ wr
        secondary = self.weighted_inv(J_s) @ xrdot_s

        # TODO: tertiary task (positioning), quaternary task (nominal arm configuration)

        # Perform the inverse kinematics to get the desired joint angles and velocities
        qddot = primary + secondary # + tertiary + quaternary
        qd_last = np.concatenate((self.qd_1, self.qd_2))
        qd = qd_last + dt*qddot

        return (qd, qddot, ptip_2, n)


    def weighted_inv(self, J):
        ''' Compute the weighted inverse of J '''
        return np.linalg.pinv(J.T @ J + self.gam**2 * np.eye(self.N_COMBINED)) @ J.T


    def nullspace(self, J): 
        ''' Get the nullspace projection of J, using a weighted inverse '''
        return np.eye(self.N_COMBINED) - self.weighted_inv(J) @ J


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