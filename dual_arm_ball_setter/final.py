# Import the format for the condition number message
from std_msgs.msg import Float64

import rclpy
import numpy as np
from scipy.optimize import fsolve

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
        self.TABLE_HEIGHT = 1.5

        ##################### ROBOT INITIALIZATION #####################
        # Set up the kinematic chain objects
        self.chain_1 = KinematicChain(node, 'base', 'panda_1_hand', self.jointnames_1())
        self.chain_2 = KinematicChain(node, 'base', 'panda_2_paddle', self.jointnames_2())
        self.chain_to_hand_2 = KinematicChain(node, 'base', 'panda_2_hand', self.jointnames_2())

        self.Jblank = np.zeros((3, self.N_COMBINED))

        indexlist = lambda start, n: list(range(start, start+n))
        self.i_1 = indexlist(0, self.N_ARM)
        self.i_2 = indexlist(self.N_ARM, self.N_ARM)

        self.qd_1 = np.array([1.37824081, 1.01533737, -0.11654914, 1.55366408, 0.10457677, 0.83709106, 0.91701575])
        self.qd_2 = np.array([0.98413218, 0.85909937, -2.66924646, -2.16988993, -2.70452348, 3.13046597, -2.41125022])
        self.qddot_1 = np.radians(np.zeros(self.N_ARM))
        self.qddot_2 = np.radians(np.zeros(self.N_ARM))
        (p0_2, R0_2, _, _) = self.chain_2.fkin(self.qd_2)
        self.p0_2 = p0_2

        self.lam = 20
        self.lam_q = 10  # set lower lamda for q, so it doesn't interfere with the other tasks
        self.gam = 0.1

        # Desired position and orientaion of chain1 tip in coordinates of and relative to chain2 tip
        self.pd_12 = np.array([0.0, 0.0, self.PADDLE_RADIUS])  
        self.Rd_12 = Rotx(np.pi)

        ##################### BALL INITIALIZATION #####################
        self.g = 2

        p0_ball = np.array([random.uniform(0.0, 3.0), random.uniform(-0.5, 0.5), random.uniform(0.0, 10.0)])
        v0_ball = np.array([random.uniform(-0.5, 0.5), random.uniform(-1.0, 1.0), random.uniform(1.0, 2.0)])

        p0_ball = np.array([1.5, 0.0, 1.5])
        v0_ball = np.array([-1.0, 0.0, 1.25])
        #p0_ball = np.array([0.0, 0.0, 6.0])
        #v0_ball = np.array([0.0, 0.0, 0.5])

        (self.x_c, self.y_c, self.z_c) = self.p0_2
        (self.x_0, self.y_0, self.z_0) = p0_ball
        (self.v0_x, self.v0_y, self.v0_z) = v0_ball

        initial_guess = 100.0
        self.T = fsolve(self.derivative_dist_paddle_to_traj, initial_guess)[0]
        self.pball_final = self.compute_ball_kin(self.T)

        # Reject solutions below table
        if self.pball_final[2] < self.TABLE_HEIGHT:
            self.T = max(np.roots([-1/2*self.g, self.v0_z, self.z_0 - self.TABLE_HEIGHT]))
            self.pball_final = self.compute_ball_kin(self.T)

        self.vball_impact = np.array([self.v0_x, self.v0_y, self.v0_z - self.g*self.T])

        self.pball = p0_ball
        self.vball = v0_ball
        self.aball = np.array([0.0, 0.0, -self.g])

        self.n0 = R0_2[:, 2]
        self.nf = -self.vball_impact / np.linalg.norm(self.vball_impact)


    def compute_ball_kin(self, t):
        return np.array([self.x_0 + self.v0_x * self.T,
                         self.y_0 + self.v0_y * self.T,
                         self.z_0 + self.v0_z * self.T - 1/2*self.g*self.T**2])


    def derivative_dist_paddle_to_traj(self, t):
        x_comp = 2 * (self.x_0 + self.v0_x * t - self.x_c) * self.v0_x
        y_comp = 2 * (self.y_0 + self.v0_y * t - self.y_c) * self.v0_y
        z_comp = 2 * (self.z_0 + self.v0_z * t - 1/2 * self.g * t ** 2 - self.z_c) * (self.v0_z - self.g * t)
        return x_comp + y_comp + z_comp
     

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
        # Trajectory generation
        scale = 1/3
        if t < self.T * scale:
            (pd, vd) = goto(t, self.T * scale, self.p0_2, self.pball_final)

            theta = acos(np.dot(self.n0, self.nf))
            (alpha, alpha_dot) = goto(t, self.T * scale, 0.0, theta)
            a_hat = np.cross(self.n0, self.nf) / sin(theta)
            nd = Rotn(a_hat, alpha) @ self.n0
            wd = alpha_dot * a_hat
        else:
            pd = self.pball_final
            vd = np.zeros(3)
            nd = -self.vball_impact / np.linalg.norm(self.vball_impact)
            wd = np.zeros(3)

        # Compute inverse kinematics
        (qd, qddot, ptip_2, n) = self.ikin(dt, wd, nd, pd, vd)
        self.qd_1 = qd[self.i_1]
        self.qd_2 = qd[self.i_2]

        # Concatenate zeros for unused finger joints
        qd = np.concatenate((qd, np.zeros(4)))
        qddot = np.concatenate((qddot, np.zeros(4)))
        
        # Update ball motion
        self.pball += dt * self.vball
        self.vball += dt * self.aball

        # Determine if the ball is in collision with the paddle
        r = self.pball - ptip_2
        if n @ r < 1e-2 and np.linalg.norm(r) <= self.PADDLE_RADIUS:
            v_normal = (self.vball @ n) * n
            v_plane = self.vball - v_normal
            self.vball = v_plane - v_normal
        # Otherwise, handle collision with table top
        elif -0.5 <= self.pball[0] <= 0.5 and -1 <= self.pball[1] < 1 and self.pball[2] <= 1:
            self.vball[2] *= -1
        # Otherwise, handle collision with floor
        elif self.pball[2] <= 0.0:
            self.vball[2] *= -1

        return (qd, qddot, self.pball)


    def ikin(self, dt, wd, nd, pd, vd):
        ''' Compute the inverse kinematics of the combined 14 DOF system
            to achieve the desired primary, secondary, tertiary, and quaternary tasks '''

        # Calculate the individual chains
        (ptip_1, Rtip_1, Jvbar_1, Jwbar_1) = self.chain_1.fkin(self.qd_1)
        (ptip_2, Rtip_2, Jvbar_2, Jwbar_2) = self.chain_2.fkin(self.qd_2)
        qd_last = np.concatenate((self.qd_1, self.qd_2))

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

        # Primary task -- keeping hands together
        J_p = np.vstack((Jv_12, Jw_12))
        ep_12 = ep(self.pd_12, p_12)
        eR_12 = eR(self.Rd_12, R_12)
        error_p = np.concatenate((ep_12, eR_12))
        xddot_p = np.zeros(6)
        qdot_p = self.weighted_inv(J_p) @ (xddot_p + self.lam*error_p)

        # Secondary task -- normal
        n = Rtip_2[:,0]            # Get unit normal vector to paddle surface (x-basis vector)
        A = np.array([[0, 1, 0], [0, 0, 1]])
        J_s = A @ Rtip_2.T @ Jw_2  # Define J_s to ignore rotation about n (partial orientation Jacobian)
        en = np.cross(n, nd)
        wr = (wd + self.lam*en)
        xrdot_s = A @ Rtip_2.T @ wr
        qdot_s = self.weighted_inv(J_s) @ xrdot_s

        # Tertiary task -- position
        J_t = Jv_2 
        e_pos = ep(pd, ptip_2)
        xrdot_t = (vd + self.lam*e_pos)
        qdot_t = self.weighted_inv(J_t) @ xrdot_t

        # Quaternary task -- natural arm configuration
        qdot_q = np.zeros(14)
        qdot_q[3] = -np.pi/2 - qd_last[3]
        qdot_q[10] = -np.pi/2 - qd_last[10]
        qdot_q[6] = np.pi/4 - qd_last[6]
        qdot_q[13] = np.pi/4 - qd_last[13]
        qdot_q *= self.lam_q 

        # Perform the inverse kinematics to get the desired joint angles and velocities
        qdot_extra = self.nullspace(J_p) @ qdot_s
        qdot_extra += self.nullspace(np.vstack((J_p, J_s))) @ qdot_t
        qdot_extra += self.nullspace(np.vstack((J_p, J_s, J_t))) @ qdot_q
        qddot = qdot_p + qdot_extra
        qd = qd_last + dt*qddot

        print("PRIMARY COST: ", np.linalg.norm(error_p))
        print("SECONDARY COST: ", np.linalg.norm(en))
        print("TERTRIARY COST: ", np.linalg.norm(e_pos))
        print("QUATERNARY COST: ", np.linalg.norm(qdot_q))
        print('---------------------------------')

        return (qd, qddot, ptip_2, n)


    def weighted_inv(self, J):
        ''' Compute the weighted inverse of J '''
        N = (J.T @ J).shape[0]
        return np.linalg.pinv(J.T @ J + self.gam**2 * np.eye(N)) @ J.T


    def nullspace(self, J): 
        ''' Get the nullspace projection of J, using a weighted inverse '''
        N = (self.weighted_inv(J) @ J).shape[0]
        return np.eye(N) - self.weighted_inv(J) @ J


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