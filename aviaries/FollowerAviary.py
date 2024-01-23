from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from trajectories import TrajectoryFactory, DiscretizedTrajectory, Waypoint

class FollowerAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 target_trajectory: DiscretizedTrajectory,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs: np.ndarray = np.array([[0.,     0.,     0.1125]]),
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        
        self.EPISODE_LEN_SEC = 16
        self.NUM_DRONES = 1
        self.WAYPOINT_BUFFER_SIZE = 3 # how many steps into future to interpolate

        self.INIT_XYZS = initial_xyzs
        self.trajectory = target_trajectory
        self.TARGET_POS = self.trajectory[len(self.trajectory) - 1]

        self.n_waypoints = len(self.trajectory)
        self.current_waypoint_idx = 0

        self.waypoint_buffer = np.array(
            [self.trajectory[i].coordinate for i in range(self.WAYPOINT_BUFFER_SIZE)]
        )
        print('WAYPOINT_BUFFER')
        print(self.waypoint_buffer)

        super().__init__(
            drone_model=drone_model,
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


    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        alpha = .5
        beta = 0.0025

        state = self._getDroneStateVector(0)
        waypoint = self.waypoint_buffer[self.current_waypoint_idx]
        reached_pos_reward = 1 if np.linalg.norm(waypoint-state[0:3]) < 0.001 else 0
        distance_reward = max(0, 2 - np.linalg.norm(waypoint-state[0:3])**4)
        distance_reward = alpha * distance_reward
        return alpha * reached_pos_reward + beta * distance_reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS.coordinate-state[0:3]) < .0001:
            return True
        else:
            return False
        
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
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

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
            # OBS SPACE OF SIZE 12
            # Observation vector - X Y Z Q1 Q2 Q3 Q4 R P Y VX VY VZ WX WY WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo]])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi]])
            #Add action buffer to observation space
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo]])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi]])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo]])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi]])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo]])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi]])])

            # Add future waypoints to observation space
            obs_lower_bound = np.hstack([obs_lower_bound, np.array([[lo,lo,lo] for i in range(self.WAYPOINT_BUFFER_SIZE)]).reshape(1, -1)])
            obs_upper_bound = np.hstack([obs_upper_bound, np.array([[hi,hi,hi] for i in range(self.WAYPOINT_BUFFER_SIZE)]).reshape(1, -1)])

            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def update_waypoints(self):
        drone_position = self._getDroneStateVector(0)[0:3]
        current_waypoint = self.waypoint_buffer[self.current_waypoint_idx]
        if np.linalg.norm(drone_position - current_waypoint) < .001:
            # replace reached waypoint with the waypoint that follows after all waypoints in the buffer
            next_waypoint_idx = int(self.current_waypoint_idx + len(self.waypoint_buffer)) % len(self.trajectory)
            next_waypoint = self.trajectory[next_waypoint_idx].coordinate
            self.waypoint_buffer[self.current_waypoint_idx] = next_waypoint
            # set next waypoint
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % self.WAYPOINT_BUFFER_SIZE


    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :]]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array(self.action_buffer[i])])
            
            #### Add future waypoints to observation
            self.update_waypoints()
            ret = np.hstack([ret, self.waypoint_buffer.reshape(1, -1)])
            return ret
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
