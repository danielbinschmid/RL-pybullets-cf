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

        # Q=diag(MX([100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        Q = diagcat(100, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        # R=diag(MX([10.0, 10.0, 10.0, 10.0]))
        R = diagcat(10.0, 10.0, 10.0, 10.0)

        x_init = P[0:Nx]
        g = X[:, 0] - P[0:Nx]  # initial condition constraints
        h = 0.2
        J = 0

        for k in range(Nhoriz - 1):
            st_ref = P[Nx:2 * Nx]
            st = X[:, k]
            cont = U[:, k]
            J += (st - st_ref).T @ Q @ (st - st_ref) + cont.T @ R @ cont
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

        thrust_step_0 = u[0, 0]  # Thrust
        tau_phi_step_0 = u[1, 0]  # Tau_phi
        tau_theta_step_0 = u[2, 0]  # Tau_theta
        tau_psi_step_0 = u[3, 0]  # Tau_psi

        '''u_np = np.array(u.full())
        np.set_printoptions(precision=4, suppress=True)
        # print(u_np)'''

        # print("Control Inputs (u) at this iteration:")
        # print(tabulate(u_np, tablefmt="fancy_grid", showindex="always", headers=[f"Step {i}" for i in range(Nhoriz)]))

        self.X0 = reshape(sol['x'][: Nx * (Nhoriz + 1)], Nx, Nhoriz + 1)

        self.cat_states = np.dstack((
            self.cat_states,
            DM2Arr(self.X0)
        ))

        self.cat_controls = np.vstack((
            self.cat_controls,
            DM2Arr(u[:, 0])
        ))

        '''t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(h, t0, state_init, u, f)'''

        # print(X0)
        self.X0 = horzcat(
            self.X0[:, 1:],
            reshape(self.X0[:, -1], -1, 1)
        )

        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        if target_thrust is None:
            rpm = self._dslPIDAttitudeControl(control_timestep,
                                              thrust,
                                              cur_quat,
                                              computed_target_rpy,
                                              target_rpy_rates
                                              )
        else:
            # attitude control
            target_thrust = (target_thrust - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
            rpm = self._dslPIDAttitudeControl(
                control_timestep=control_timestep,
                thrust=target_thrust,
                cur_quat=cur_quat,
                target_euler=np.array(cur_rpy),
                target_rpy_rates=target_rpy_rates
            )

        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

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
