from casadi import *
from time import time
import numpy as np
# from tabulate import tabulate

import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


def DM2Arr(dm):
    return np.array(dm.full())


class MPCCasADiControl(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu,
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        self.state_target = None
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        '''
        self.Ix = 0.0000166  
        self.Iy = 0.0000167
        self.Iz = 0.00000293
        self.mass = 0.029
        self.Nx = 12
        self.Nu = 4
        self.Nhoriz = 10

        # x(t) = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]T
        self.x = MX.sym("x", self.Nx)
        self.u = MX.sym("u", self.Nu)

        self.x_pos = self.x[0]  # x-position
        self.y = self.x[1]  # y-position
        self.z = self.x[2]  # z-position
        self.phi = self.x[3]  # phi-angle, Euler angles
        self.theta = self.x[4]  # theta-angle, Euler angles
        self.psi = self.x[5]  # psi-angle, Euler angles
        self.x_pos_dot = self.x[6]  # x velocity
        self.y_dot = self.x[7]  # y velocity
        self.z_dot = self.x[8]  # z velocity
        self.phi_dot = self.x[9]  # phi_dot, angular velocity
        self.theta_dot = self.x[10]  # theta_dot
        self.psi_dot = self.x[11]  # psi-dot

        self.thrust = self.u[0]
        self.tau_phi = self.u[1]
        self.tau_theta = self.u[2]
        self.tau_psi = self.u[3]

        x_pos_ddot = (cos(self.phi) * sin(self.theta) * cos(self.psi) + sin(self.phi) * sin(self.psi)) * self.thrust / self.mass
        y_ddot = (cos(self.phi) * sin(self.theta) * cos(self.psi) - sin(self.phi) * cos(self.psi)) * self.thrust / self.mass
        z_ddot = -g + (cos(self.phi) * cos(self.theta)) * self.thrust / self.mass
        phi_ddot = self.theta_dot * self.psi_dot * (self.Iy - self.Iz) / (self.Ix) + self.tau_phi / self.Ix
        theta_ddot = self.phi_dot * self.psi_dot * (self.Iz - self.Ix) / (self.Iy) + self.tau_theta / self.Iy
        psi_ddot = self.theta_dot * self.phi_dot * (self.Ix - self.Iy) / (self.Iz) + self.tau_psi / self.Iz

        x_dot = vertcat(self.x_pos_dot, self.y_dot, self.z_dot, self.phi_dot, self.theta_dot, self.psi_dot, x_pos_ddot, y_ddot, z_ddot, phi_ddot,
                        theta_ddot, psi_ddot)
        self.f = Function('f', [self.x, self.u], [x_dot], ['x', 'u'], ['x_dot'])          
        '''

        Ix = 0.0000166  # Moment of inertia around p_WB_W_x-axis, source: Julian Förster's ETH Bachelor Thesis
        Iy = 0.0000167  # Moment of inertia around p_WB_W_y-axis, source: Julian Förster's ETH Bachelor Thesis
        Iz = 0.00000293  # Moment of inertia around p_WB_W_z-axis, source: Julian Förster's ETH Bachelor Thesis
        m = 0.029  # mass of Crazyflie 2.1
        g = 9.81

        Nx = 12
        Nu = 4
        Nhoriz = 10

        # x(t) = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, ˙theta_dot, psi_dot]T
        x = MX.sym("x", Nx)
        u = MX.sym("u", Nu)

        x_pos = x[0]  # x-position
        y = x[1]  # y-position
        z = x[2]  # z-position
        phi = x[3]  # phi-angle, Euler angles
        theta = x[4]  # theta-angle, Euler angles
        psi = x[5]  # psi-angle, Euler angles
        x_pos_dot = x[6]  # x velocity
        y_dot = x[7]  # y velocity
        z_dot = x[8]  # z velocity
        phi_dot = x[9]  # phi_dot, angular velocity
        theta_dot = x[10]  # theta_dot
        psi_dot = x[11]  # psi-dot

        thrust = u[0]
        tau_phi = u[1]
        tau_theta = u[2]
        tau_psi = u[3]

        # x_dot(t) = [x_pos_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot, x_pos_ddot, y_ddot, z_ddot, phi_ddot,
        # theta_ddot, psi_ddot]T

        x_pos_ddot = (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) * thrust / m
        y_ddot = (cos(phi) * sin(theta) * cos(psi) - sin(phi) * cos(psi)) * thrust / m
        z_ddot = -g + (cos(phi) * cos(theta)) * thrust / m
        phi_ddot = theta_dot * psi_dot * (Iy - Iz) / Ix + tau_phi / Ix
        theta_ddot = phi_dot * psi_dot * (Iz - Ix) / Iy + tau_theta / Iy
        psi_ddot = theta_dot * phi_dot * (Ix - Iy) / Iz + tau_psi / Iz

        x_dot = vertcat(x_pos_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot, x_pos_ddot, y_ddot, z_ddot, phi_ddot,
                        theta_ddot, psi_ddot)

        self.f = Function('f', [x, u], [x_dot], ['x', 'u'], ['x_dot'])
        f = Function('f', [x, u], [x_dot], ['x', 'u'], ['x_dot'])

        U = MX.sym("U", Nu, Nhoriz)  # Decision variables (controls)
        P = MX.sym('P', Nx + Nx)  # parameters (which include the initial state and the reference state)
        X = MX.sym("X", Nx, Nhoriz + 1)  # A vector that represents the states over the optimization problem.

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([
                [-.5, -.5, -1],
                [-.5, .5, 1],
                [.5, .5, -1],
                [.5, -.5, 1]
            ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                [0, -1, -1],
                [+1, 0, 1],
                [0, 1, -1],
                [-1, 0, 1]
            ])

        J = 0  # Objective function
        g = []  # constraints vector

        ### Weighting Matrices Q and R for objective function
        ### 20 tracks test suite experiments and test results with varying trajectory discretization levels

        ##### PID reference performance on the 20 tracks test suite

        ### Stop and Go
        #####################             if distance < 0.2 and velocity < 0.05: # Original Stop-And-GO
            # if current_step == len(TARGET_TRAJECTORY) - 1 and velocity < 1.0: #####################
        # N DISCR LEVEL: 10
        # COMPLETION TIME MEAN: 8.832291666666666
        # SUCCESS RATE: 1.0
        # AVERAGE DEVIATION:  0.05192337496578584
        # MAXIMUM DEVIATION: 0.14372815993134694

        ##################### distance < 0.1 #####################
        # DEFAULT_DISCR_LEVEL = 10
        # -> misses the end goal at one track

        # Generally with this distance it does get out of the loop

        ##################### if distance < 0.2 and velocity < 1.0:
        #    if current_step == len(TARGET_TRAJECTORY) - 1 and velocity < 1.0: #####################

        # DEFAULT_DISCR_LEVEL = 10
        # COMPLETION TIME MEAN: 4.537280701754386 (+ 1 Second and minus some accuracy, because it terminates quite early)
        # SUCCESS RATE: 0.95 -> It actually fails in the ninth track
        # AVERAGE DEVIATION:  0.06635711385416021
        # MAXIMUM DEVIATION: 0.1806313058800846

        ##################### distance < 0.05 #####################

        # DEFAULT_DISCR_LEVEL = 20
        # Hangs up itself because of small distance

        # N DISCR LEVEL: 30
        # -> misses the second track

        ################################################################################################################

        '''# My own designed matrices partially inspired from TinyMPC
        #Q = diagcat(100, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        #R = diagcat(10.0, 10.0, 10.0, 10.0)
        
        # Normalized (Does this make a difference?) - Position set to 1
        #Q = diagcat(1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
        #R = diagcat(0.1, 0.1, 0.1, 0.1)
        # DEFAULT_DISCR_LEVEL = 50 -> It seems to work, but looks too rigid and too slow'''


        '''# Matrices taken from somewhere, maybe TinyMPC, can't remember
        Q=diag(MX([100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        R=diag(MX([10.0, 10.0, 10.0, 10.0]))
        
        # Normalized (Does this make a difference?) - Position set to 1
        #Q = diagcat(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        #R = diagcat(0.1, 0.1, 0.1, 0.1)
        # DEFAULT_DISCR_LEVEL = 50 -> Completely goes off rails at the beginning'''


        '''# Paper: Non-Linear Model Predictive Control Using CasADi Package for Trajectory Tracking of Quadrotor
        # The weighting matrix Q = Diag[1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0],
        # while the control input weighting matrix R = Diag[0.3, 0.3, 0.3, 0.8]
        
        Q = diagcat(1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.3, 0.3, 0.3, 0.8)

        ##################### distance < 0.05 #####################

        # DEFAULT_DISCR_LEVEL = 10 ->
        # 5%|▌         | 1/20 [01:04<20:33, 64.94s/it]
        # 10%|█         | 2/20 [02:14<20:16, 67.61s/it]
        # 3rd track: Misses the last point by a bit

        # DEFAULT_DISCR_LEVEL = 15 ->
        # First try
            # 5%|▌         | 1/20 [01:23<26:24, 83.38s/it]
            # 10%|█         | 2/20 [03:08<28:20, 94.49s/it]
        #Second try
            # 5%|▌         | 1/20 [01:19<25:17, 79.86s/it]
            # 10%|█         | 2/20 [02:43<24:39, 82.20s/it]
            # 3rd track: Misses the last point by quite a bit

        # DEFAULT_DISCR_LEVEL = 20 ->
        # First try
            # 5%|▌         | 1/20 [01:47<33:59, 107.33s/it]
        # Second try
            # 5%|▌         | 1/20 [01:45<33:24, 105.48s/it]
            # 10%|█         | 2/20 [03:12<28:22, 94.56s/it]
            # 15%|█▌        | 3/20 [05:44<34:10, 120.63s/it]
            # 20%|██        | 4/20 [07:55<33:15, 124.71s/it]
            # N DISCR LEVEL: 20
            # COMPLETION TIME MEAN: 13.834375
            # SUCCESS RATE: 1.0
            # AVERAGE DEVIATION:  0.040151234220999796
            # MAXIMUM DEVIATION: 0.10004746548338775

        
        # DEFAULT_DISCR_LEVEL = 30 ->
        # First try
            # First track: 5%|▌         | 1/20 [02:09<40:56, 129.26s/it]
            # 10%|█         | 2/20 [03:58<35:11, 117.32s/it]finished
            # 15%|█▌        | 3/20 [05:57<33:30, 118.25s/it]finished
            #COMPLETION TIME MEAN: 15.638157894736842
            #SUCCESS RATE: 0.95
            #AVERAGE DEVIATION:  0.029817046088798104
            #MAXIMUM DEVIATION: 0.06448328287707432

            # DEFAULT_DISCR_LEVEL = 50 ->
            # First track: 5%|▌         | 1/20 [03:49<1:12:48, 229.91s/it]
        # Second try
            # 5%|▌         | 1/20 [02:16<43:10, 136.33s/it]
            # 10%|█         | 2/20 [04:10<37:04, 123.58s/it]
            # 15%|█▌        | 3/20 [06:32<37:19, 131.74s/it]
            # 20%|██        | 4/20 [09:32<40:12, 150.75s/it]'''


        '''# Only care greedily about the position 
        Q = diagcat(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        R = diagcat(0, 0, 0, 0)
        # -> Test result: Drone completely goes off the rails and crazyflie flies crazily and fails'''


        '''# Don't care about the velocity of the drone
        Q = diagcat(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.3, 0.3, 0.3, 0.8)
        # -> Test result: Drone completely goes off the rails and crazyflie flies crazily and fails'''

        '''# Don't care about the reference input of the drone
        Q = diagcat(1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0)
        R = diagcat(0, 0, 0, 0)
        # -> Test result: Drone circles around a reference point indefinitely and doesn't reach it'''

        '''# Care 6x (Correction: was 3x) less about  and velocities and do not care about the reference input of the drone
        Q = diagcat(1, 1, 1, 0.1, 0.1, 0.1666, 0, 0, 0, 0, 0, 0)
        R = diagcat(0, 0, 0, 0)
        # -> Test result:
        # DEFAULT_DISCR_LEVEL = 10 -> goes off track after 1 point
        # DEFAULT_DISCR_LEVEL = 20 -> goes off track after 2 points
        # DEFAULT_DISCR_LEVEL = 30 -> goes off track after 7-8 points
        # DEFAULT_DISCR_LEVEL = 40 -> goes off track after 4 points
        # DEFAULT_DISCR_LEVEL = 50 -> goes off track after 4 points'''

        '''# Care 3x less about and velocities and reference input of the drone
            # Reference Weighting matrices from the paper
            #Q = diagcat(1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0)
            #R = diagcat(0.3, 0.3, 0.3, 0.8)

        Q = diagcat(1, 1, 1, 0.1, 0.1, 0.1666, 0, 0, 0, 0, 0, 0) #!! Correction this is imbalanced weighting! This is 6x less
        # It should be Q = diagcat(1, 1, 1, 0.2, 0.2, 0.1666, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.1, 0.1, 0.1, 0.2666) #
        # DEFAULT_DISCR_LEVEL = 10 -> goes off track after 1 points
        # DEFAULT_DISCR_LEVEL = 30 -> goes off track after 10 points
        # DEFAULT_DISCR_LEVEL = 50 -> goes off track after 4 points

        # ---> The correction:
        # Care 3x less about and velocities and reference input of the drone
        Q = diagcat(1, 1, 1, 0.2, 0.2, 0.1666, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.1, 0.1, 0.1, 0.2666)
        # DEFAULT_DISCR_LEVEL = 10 -> goes off track after 3 points
        # DEFAULT_DISCR_LEVEL = 30 -> finishes 1st track with lots of mistakes
        # DEFAULT_DISCR_LEVEL = 50 -> Performs actually quite well with minor mistakes
        # DEFAULT_DISCR_LEVEL = 100 -> Performs actually quite well, but goes off track midway
        # DEFAULT_DISCR_LEVEL = 200 -> Performs actually quite well with minor mistakes. A little slower. Misses
        # the last goal position (0,0,0) though'''


        '''# Care 2x less about and velocities and reference input of the drone
            # Reference Weighting matrices from the paper
            # Q = diagcat(1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0)
            # R = diagcat(0.3, 0.3, 0.3, 0.8)

        Q = diagcat(1, 1, 1, 0.3, 0.3, 0.2, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.15, 0.15, 0.15, 0.4)

        # DEFAULT_DISCR_LEVEL = 50 -> Performs actually well and finishes tracks, but is less accurate at some situations
        # and misses points and has to go back
        # DEFAULT_DISCR_LEVEL = 100 -> Quite slow 5%|▌         | 1/20 [03:38<1:09:19, 218.93s/it]'''

        '''# Care only 75% less about and velocities and reference input of the drone
            # Reference Weighting matrices from the paper
            # Q = diagcat(1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0)
            # R = diagcat(0.3, 0.3, 0.3, 0.8)

        Q = diagcat(1, 1, 1, 0.45, 0.45, 0.75, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.225, 0.225, 0.225, 0.6)

        # DEFAULT_DISCR_LEVEL = 30 -> Good, but still worse than Paper weights
        # 5%|▌         | 1/20 [02:10<41:15, 130.31s/it]
        # 10%|█         | 2/20 [04:11<37:31, 125.09s/it]
        # 15%|█▌        | 3/20 [06:41<38:39, 136.46s/it]
        # DEFAULT_DISCR_LEVEL = 100 ->  Quite slow
        # 5%|▌         | 1/20 [04:23<1:23:31, 263.79s/it]
        # 10%|█         | 2/20 [07:45<1:08:12, 227.35s/it]'''

        ################################################################################################################

        #####################if distance < 0.2 and velocity < 1.0:
        #    if current_step == len(TARGET_TRAJECTORY) - 1 and velocity < 1.0: #####################

        '''# Paper: Non-Linear Model Predictive Control Using CasADi Package for Trajectory Tracking of Quadrotor
        # The weighting matrix Q = Diag[1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0],
        # while the control input weighting matrix R = Diag[0.3, 0.3, 0.3, 0.8]

        Q = diagcat(1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.3, 0.3, 0.3, 0.8)
        
        # N DISCR LEVEL: 10
        # COMPLETION TIME MEAN: 4.672916666666667
        # SUCCESS RATE: 1.0
        # AVERAGE DEVIATION:  0.10109089137096094
        # MAXIMUM DEVIATION: 0.22336670663036812'''

        '''# Only care greedily about the position
        Q = diagcat(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        R = diagcat(0, 0, 0, 0)
        # -> Test result: Drone completely goes off the rails and crazyflie flies crazily and fails'''

        '''# Care only 62.5% less about and velocities and reference input of the drone
                    # Reference Weighting matrices from the paper
                    # Q = diagcat(1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0)
                    # R = diagcat(0.3, 0.3, 0.3, 0.8)

        Q = diagcat(1, 1, 1, 0.375, 0.375, 0.625, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.1875, 0.1875, 0.1875, 0.5)
        
        # DEFAULT_DISCR_LEVEL = 10 ->
        # First track only
        # COMPLETION TIME MEAN: 3.9583333333333335
        # SUCCESS RATE: 1.0
        # AVERAGE DEVIATION:  0.07690801583058962
        # MAXIMUM DEVIATION: 0.14062837262156286 '''


        # Care 2x less (or only 50%) about and velocities and reference input of the drone
                    # Reference Weighting matrices from the paper
                    # Q = diagcat(1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0)
                    # R = diagcat(0.3, 0.3, 0.3, 0.8)

        Q = diagcat(1, 1, 1, 0.3, 0.3, 0.2, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.15, 0.15, 0.15, 0.4)

        ###             if distance < 0.2:
                # if current_step == len(TARGET_TRAJECTORY) - 1:

        # N DISCR LEVEL: 10
        # COMPLETION TIME MEAN: 4.119791666666666
        # SUCCESS RATE: 1.0
        # AVERAGE DEVIATION:  0.09995147199727226
        # MAXIMUM DEVIATION: 0.2386860064898518

        '''# Care (or only  42.5%) about and velocities and reference input of the drone
                            # Reference Weighting matrices from the paper
                            # Q = diagcat(1, 1, 1, 0.6, 0.6, 1, 0, 0, 0, 0, 0, 0)
                            # R = diagcat(0.3, 0.3, 0.3, 0.8)

        Q = diagcat(1, 1, 1, 0.255, 0.255, 0.425, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.1275, 0.1275, 0.1275, 0.34)

                ###             if distance < 0.2:
                        # if current_step == len(TARGET_TRAJECTORY) - 1:

                # N DISCR LEVEL: 10 -> performs a bit slower than 50%'''



        '''# Care 3x less about (or only 33%) and velocities and reference input of the drone
        Q = diagcat(1, 1, 1, 0.2, 0.2, 0.1666, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.1, 0.1, 0.1, 0.2666)
        
        # DEFAULT_DISCR_LEVEL = 10 -> causes it to fail on at least two tracks midways'''



        '''# Don't care about the velocity of the drone
        Q = diagcat(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        R = diagcat(0.3, 0.3, 0.3, 0.8)
        # -> Test result: Drone completely goes off the rails and crazyflie flies crazily and fails'''

        x_init = P[0:Nx]
        g = X[:, 0] - P[0:Nx]  # initial condition constraints
        h = 0.15
        J = 0

        # Calculate the hovering thrust as a scalar
        hovering_thrust = 0.28449  # 0.029 * 9.81   m and g should be scalar values (floats or ints)
        tau_phi_ref = 0
        tau_theta_ref = 0
        tau_psi_ref = 0

        # Create a DM vector for the reference control input
        u_ref = DM([hovering_thrust, tau_phi_ref, tau_theta_ref, tau_psi_ref])

        for k in range(Nhoriz - 1):
            st_ref = P[Nx:2 * Nx]
            st = X[:, k]
            cont = U[:, k]
            cont_ref = u_ref
            J += (st - st_ref).T @ Q @ (st - st_ref) + (cont - cont_ref).T @ R @ (cont - cont_ref)
            st_next = X[:, k + 1]
            k1 = f(st, cont)
            k2 = f(st + h / 2 * k1, cont)
            k3 = f(st + h / 2 * k2, cont)
            k4 = f(st + h * k3, cont)
            st_next_RK4 = st + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # RK4 integration
            g = vertcat(g, st_next - st_next_RK4)  # Multiple Shooting

        OPT_variables = vertcat(
            X.reshape((-1, 1)),  # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            U.reshape((-1, 1))
        )

        nlp_prob = {
            'f': J,
            'x': OPT_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'max_iter': 20,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        self.solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

        lbx = DM.zeros((Nx * (Nhoriz + 1) + Nu * Nhoriz, 1))
        ubx = DM.zeros((Nx * (Nhoriz + 1) + Nu * Nhoriz, 1))

        lbx[0: Nx * (Nhoriz + 1): Nx] = -inf  # x lower bound
        lbx[1: Nx * (Nhoriz + 1): Nx] = -inf  # y lower bound
        lbx[2: Nx * (Nhoriz + 1): Nx] = -inf  # z lower bound
        lbx[3: Nx * (Nhoriz + 1): Nx] = -inf  # phi lower bound
        lbx[4: Nx * (Nhoriz + 1): Nx] = -inf  # theta lower bound
        lbx[5: Nx * (Nhoriz + 1): Nx] = -inf  # psi lower bound
        lbx[6: Nx * (Nhoriz + 1): Nx] = -inf  # x_dot lower bound
        lbx[7: Nx * (Nhoriz + 1): Nx] = -inf  # y_dot lower bound
        lbx[8: Nx * (Nhoriz + 1): Nx] = -inf  # z_dot lower bound
        lbx[9: Nx * (Nhoriz + 1): Nx] = -inf  # phi_dot lower bound
        lbx[10: Nx * (Nhoriz + 1): Nx] = -inf  # theta_dot lower bound
        lbx[11: Nx * (Nhoriz + 1): Nx] = -inf  # psi_dot lower bound

        ubx[0: Nx * (Nhoriz + 1): Nx] = inf  # x upper bound
        ubx[1: Nx * (Nhoriz + 1): Nx] = inf  # y upper bound
        ubx[2: Nx * (Nhoriz + 1): Nx] = inf  # z upper bound
        ubx[3: Nx * (Nhoriz + 1): Nx] = inf  # phi upper bound
        ubx[4: Nx * (Nhoriz + 1): Nx] = inf  # theta upper bound
        ubx[5: Nx * (Nhoriz + 1): Nx] = inf  # psi upper bound
        ubx[6: Nx * (Nhoriz + 1): Nx] = inf  # x_dot upper bound
        ubx[7: Nx * (Nhoriz + 1): Nx] = inf  # y_dot upper bound
        ubx[8: Nx * (Nhoriz + 1): Nx] = inf  # z_dot upper bound
        ubx[9: Nx * (Nhoriz + 1): Nx] = inf  # phi_dot upper bound
        ubx[10: Nx * (Nhoriz + 1): Nx] = inf  # theta_dot upper bound
        ubx[11: Nx * (Nhoriz + 1): Nx] = inf  # psi_dot upper bound

        lbx[Nx * (Nhoriz + 1):] = -1.2  # v lower bound for all u
        ubx[Nx * (Nhoriz + 1):] = 1.2  # v upper bound for all u

        # Starting index for the control variables section
        start_idx_control = Nx * (Nhoriz + 1)

        # Set bounds for the thrust (first control input) and tau_psi (fourth control input) across all timesteps in
        # a single loop
        for k in range(Nhoriz):
            # Index for the thrust control input at timestep k
            idx_thrust = start_idx_control + k * Nu  # First control input

            # Index for tau_psi at timestep k
            idx_tau_psi = start_idx_control + k * Nu + 3  # Fourth control input, hence +3

            # Set lower and upper bounds for thrust
            lbx[idx_thrust] = 0
            ubx[idx_thrust] = 35  # Adjusted as per your correction to 35

            # Set lower and upper bounds for tau_psi
            lbx[idx_tau_psi] = -0.2
            ubx[idx_tau_psi] = 0.2

        self.args = {
            'lbg': DM.zeros((Nx * Nhoriz, 1)),  # constraints lower bound
            'ubg': DM.zeros((Nx * Nhoriz, 1)),  # constraints upper bound
            'lbx': lbx,
            'ubx': ubx
        }

        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        Nx = 12
        Nu = 4
        Nhoriz = 10

        super().reset()

        x_pos_init = 0
        y_init = 0
        z_init = 0
        phi_init = 0  # roll
        theta_init = 0  # pitch
        psi_init = 0  # yaw
        x_pos_dot_init = 0
        y_dot_init = 0
        z_dot_init = 0
        phi_dot_init = 0
        theta_dot_init = 0
        psi_dot_init = 0

        self.state_init = DM(
            [x_pos_init, y_init, z_init, phi_init, theta_init, psi_init, x_pos_dot_init, y_dot_init, z_dot_init,
             phi_dot_init, theta_dot_init, psi_dot_init])  # initial state

        self.u0 = DM.zeros((Nu, Nhoriz))  # initial control
        self.X0 = repmat(self.state_init, 1, Nhoriz + 1)  # initial state full

        self.cat_states = DM2Arr(self.X0)
        self.cat_controls = DM2Arr(self.u0[:, 0])

        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3),
                       target_thrust: float = None,
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        Nx = 12
        Nu = 4
        Nhoriz = 10

        ## INIT
        x_pos_init = cur_pos[0]
        y_init = cur_pos[1]
        z_init = cur_pos[2]

        # Convert quaternion to Euler angles for roll, pitch, and yaw
        phi_init, theta_init, psi_init = p.getEulerFromQuaternion(cur_quat)

        x_pos_dot_init = cur_vel[0]
        y_dot_init = cur_vel[1]
        z_dot_init = cur_vel[2]

        phi_dot_init = cur_ang_vel[0]
        theta_dot_init = cur_ang_vel[1]
        psi_dot_init = cur_ang_vel[2]

        ## TARGET
        # Set position targets
        x_pos_target = target_pos[0]
        y_target = target_pos[1]
        z_target = target_pos[2]

        # Set orientation targets (roll, pitch, yaw)
        phi_target = target_rpy[0]  # roll
        theta_target = target_rpy[1]  # pitch
        psi_target = target_rpy[2]  # yaw

        # Set velocity targets
        x_pos_dot_target = target_vel[0]
        y_dot_target = target_vel[1]
        z_dot_target = target_vel[2]

        # Set angular velocity targets (roll_dot, pitch_dot, yaw_dot)
        phi_dot_target = target_rpy_rates[0]
        theta_dot_target = target_rpy_rates[1]
        psi_dot_target = target_rpy_rates[2]

        self.state_init = DM(
            [x_pos_init, y_init, z_init, phi_init, theta_init, psi_init, x_pos_dot_init, y_dot_init, z_dot_init,
             phi_dot_init, theta_dot_init, psi_dot_init])  # initial state

        self.u0 = DM.zeros((Nu, Nhoriz))  # initial control
        self.X0 = repmat(self.state_init, 1, Nhoriz + 1)  # initial state full

        self.cat_states = DM2Arr(self.X0)
        self.cat_controls = DM2Arr(self.u0[:, 0])

        self.state_target = DM(
            [x_pos_target, y_target, z_target, phi_target, theta_target, psi_target, x_pos_dot_target, y_dot_target,
             z_dot_target, phi_dot_target, theta_dot_target, psi_dot_target])  # target state

        self.args['p'] = vertcat(
            self.state_init,  # current state
            self.state_target  # target state
        )
        # optimization variable current state
        self.args['x0'] = vertcat(
            reshape(self.X0, Nx * (Nhoriz + 1), 1),
            reshape(self.u0, Nu * Nhoriz, 1)
        )

        sol = self.solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )

        u = reshape(sol['x'][Nx * (Nhoriz + 1):], Nu, Nhoriz)

        thrust_step_0 = float(u[0, 0])  # Thrust
        tau_phi_step_0 = float(u[1, 0])  # Tau_phi
        tau_theta_step_0 = float(u[2, 0])  # Tau_theta
        tau_psi_step_0 = float(u[3, 0])  # Tau_psi
        u_step_0 = u[:, 0]

        torques = np.array([
            tau_phi_step_0,
            tau_theta_step_0,
            tau_psi_step_0,
        ])

        #####

        ## Rotation information
        target_rotation = np.array([float(tau_phi_step_0), float(tau_theta_step_0), float(tau_psi_step_0)])
        computed_target_rpy = target_rotation
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)

        ## Inverse Mixer Matrix Approach
        '''M_matrix = np.array([
            [-.5, -.5, -1],
            [-.5, .5, 1],
            [.5, .5, -1],
            [.5, -.5, 1]
        ])

        x_step_0 = self.f(self.state_init, u_step_0)

        if M_matrix.shape[0] != M_matrix.shape[1]:
            print("The matrix must be square to be invertible.")
        else:
            # Compute the inverse
            M_matrix_inv = np.linalg.inv(M_matrix)
            print("Inverse of the matrix:")

        M_matrix_inv

        RPMs_squared = M_matrix_inv @ torques
        RPMs = np.sqrt(RPMs_squared)'''

        ## Model predictive angles approach
        '''phi = x_step_0[3]  # phi-angle, Euler angles
        theta = x_step_0[4]  # theta-angle, Euler angles
        psi = x_step_0[5]  # psi-angle, Euler angles
        #target_rotation = np.array([float(phi), float(theta), float(psi)])'''

        ## Just map it to the range approach

        # target_torques = np.array([float(tau_phi_step_0) * 3200 / 1.257, float(tau_theta_step_0) * 3200 / 1.257, float(tau_psi_step_0) * 3200 / 0.2145])

        ## ETH system identification functions approach. Source: Julian Förster's ETH Zurich Bachelor Thesis

        # Torque tau_i -> Thrust f_i
        TORQUES_TUNING = 1.0

        tau_i = np.array([
            tau_phi_step_0,
            tau_theta_step_0,
            tau_psi_step_0,
        ])
        f_i = 1676.57185318 * tau_i

        # Thrust f_i -> Input Command pwm_i

        pwm_i_torques = np.zeros_like(f_i, dtype=float)

        # ETH Zurich sysID : f_i = 2.130295*10^-11*pwm_i² + 1.032633*10^-6*pwm_i + 5.484560*10^-4

        # Wolfram Alpha: find the inverse function of f(x)=2.130295*10^-11*x² + 1.032633*10^-6*x+5.484560*10^-4
        # We have to make a distinction of two cases here, where f_i is positive or negative, because of the root function
        mask_pos = f_i >= 0
        mask_neg = f_i < 0

        pwm_i_torques[mask_pos] = -24236.9 + TORQUES_TUNING * 1.57508e-11 * np.sqrt(
            1.89215e32 * f_i[mask_pos] + 2.26404e30)

        # Treat the negative f_i like the f_i. We effectively mirror the root function on the y-axis
        # Then we mirror it on the x-axis to get a point-mirrored function continuation for the negative range
        pwm_i_torques[mask_neg] = (-1) * (
                    -24236.9 + TORQUES_TUNING * 1.57508e-11 * np.sqrt(1.89215e32 * -f_i[mask_neg] + 2.26404e30))

        '''### Linearization alternative
        #  f_i =2.130295*10^-11*pwm_i² + 1.032633*10^-6*pwm_i + 5.484560*10^-4 can be approximated with
        #  f_i_lin(x) = 0.15 / 65000 pwm_i =  2.30769e-6 pwm_i
        #  pwm_i_lin = 1 / 2.30769e-6 * f_i =  433333 * f_i
        pwm_i_torques = 433333 * TORQUES_TUNING * f_i'''

        # Thrust calculation

        # Original scalar thrust normalized
        THRUST_TUNING = 0.925  # h= 0.2
        thrust_normalized_scalar = THRUST_TUNING * thrust_step_0

        # Convert to a 3D vector with the thrust applied along the z-axis
        thrust_normalized = np.array([0, 0, thrust_normalized_scalar])

        # thrust = thrust_normalized * (self.MAX_PWM - self.MIN_PWM ) + self.MIN_PWM
        scalar_thrust = max(0., np.dot(thrust_normalized, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        ## PWM -> RPM

        '''# Insert a 0 at the beginning to make it 4D
        pwm_i_torques = np.insert(pwm_i_torques, 0, 0)
        pwm_i_torques = np.clip(pwm_i_torques, -3200, 3200)'''

        target_pwm = np.clip(pwm_i_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_pwm)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        '''
        Strictly speaking these are not torques anymore, but pwms:

        target_torques = np.clip(target_torques, -3200, 3200)

        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm =  self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        '''

        ####

        self.X0 = reshape(sol['x'][: Nx * (Nhoriz + 1)], Nx, Nhoriz + 1)

        self.cat_states = np.dstack((
            self.cat_states,
            DM2Arr(self.X0)
        ))

        self.cat_controls = np.vstack((
            self.cat_controls,
            DM2Arr(u[:, 0])
        ))

        self.X0 = horzcat(
            self.X0[:, 1:],
            reshape(self.X0[:, -1], -1, 1)
        )

        self.control_counter += 1

        return rpm, None, None

    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter,
                  "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e

    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(),
                                                                                    target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust) / (self.KF * (4 / DIM))) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE,
                      self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4 / DIM)
        elif DIM == 2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()