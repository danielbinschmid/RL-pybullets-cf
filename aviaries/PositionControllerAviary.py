import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from trajectories.random_trajectory import RandomTrajectory
from trajectories.trajectory import Trajectory
from typing import Dict, Any
import pybullet as p
from gymnasium import spaces


class PositionControllerAviary(BaseRLAviary):

    ################################################################################

    def __init__(self,
                 initial_xyzs: np.ndarray,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 zero_velocity_at_target: bool = False,
                 trajectory: Trajectory = RandomTrajectory()
                 ):

        self.INIT_XYZS = initial_xyzs
        self.TRAJECTORY = trajectory
        self.TARGET_POS = self.TRAJECTORY.get_next_waypoint().coordinate
        self.EPISODE_LEN_SEC = 20
        self.N_TARGET_REACHED = 0
        self.ZERO_VELOCITY_AT_TARGET = zero_velocity_at_target
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        # new target
        position = state[0:3]
        velocity = state[10:13]
        if np.linalg.norm(self.TARGET_POS-position) < .1 and np.linalg.norm(velocity) < .05:
            ret += 150
            self.N_TARGET_REACHED += 1
            waypoint = self.TRAJECTORY.get_next_waypoint()
            self.TARGET_POS = waypoint.coordinate
        return ret

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC or self.TRAJECTORY.is_done():
            return True

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self) -> Dict[str, Any]:
        """
        Any additional computation in simulation loop. 
        Returns interesting information to the user as a dictionary.
        """
        self._vis()

        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    def _vis(self):
        """
        Additional visualization in target environmnet.
        """
        # target waypoint visualization
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                radius=0.03,
                                                rgbaColor=[0, 1, 0, 1],
                                                physicsClientId=self.CLIENT)
        target = p.createMultiBody(baseMass=0.0,
                                    baseCollisionShapeIndex=-1,
                                    baseVisualShapeIndex=sphere_visual,
                                    basePosition=self.TARGET_POS,
                                    useMaximalCoordinates=False,
                                    physicsClientId=self.CLIENT)
        p.changeVisualShape(target,
                            -1,
                            rgbaColor=[0.9, 0.3, 0.3, 1],
                            physicsClientId=self.CLIENT)

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 15
            #### Observation vector ### error_X   error_Y  error_Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ, TX, TY, TZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo, lo, lo, lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi, hi, hi, hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL, ActionType.ATTITUDE_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
                else:
                    print("[ERROR] in BaseRLAviary._observationSpace()")
                    exit()
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """

        # position error, velocity and acceleration observation, and target position
        obs_12 = np.zeros((self.NUM_DRONES,15))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            obs_12[i, :12] = np.hstack([obs[0:3] - self.TARGET_POS, obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            obs_12[i, 12:15] = self.TARGET_POS

        ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')

        # action buffer
        for i in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])

        return ret

    def get_position(self, drone_id):
        return self._getDroneStateVector(drone_id)[0:3]

    def reset(self, seed: int = None, options: dict = None):
        self.TRAJECTORY.reset()
        return super().reset(seed, options)
