from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

import numpy as np
import copy
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from trajectories import TrajectoryFactory, DiscretizedTrajectory, Waypoint

class UZHAviary(BaseRLAviary):
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
        self.EPISODE_LEN_SEC = 8
        self.NUM_DRONES = 1

        self.INIT_XYZS = initial_xyzs
        self.trajectory = np.array([x.coordinate for x in target_trajectory])
        self.dist_tol = 0.08

        self.WAYPOINT_BUFFER_SIZE = 2 # how many steps into future to interpolate
        self.current_waypoint_idx = 0
        assert self.WAYPOINT_BUFFER_SIZE < len(self.trajectory), "Buffer size should be smaller than the number of waypoints"

        # pad the trajectory for waypoint buffer
        self.trajectory = np.vstack([
            self.trajectory,
            np.array(self.trajectory[-1] * np.ones((self.WAYPOINT_BUFFER_SIZE, 3)))
        ])

        # precompute trajectory variables
        self.p1 = self.trajectory[:-1]
        self.p2 = self.trajectory[1:]
        self.diffs = self.p2 - self.p1

        # waypoint distances
        self.distances = np.linalg.norm(self.p1 - self.p2, axis=1)
        self.reached_distance = 0
        

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
        self.visualised = False

    
    def reset_vars(self):
        self.current_waypoint_idx = 0
        self.reached_distance = 0

    def compute_projection(self, v, p1, p2):
        # get p1, p2 (deepcopy)
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)

        p2 -= p1
        v -= p1

        # compute projection
        coef = np.dot(v, p2) / np.dot(p2, p2)
        return coef, p2
    
    def get_travelled_distance(self):
        # track progress reward
        position = self._getDroneStateVector(0)[0:3]
        shifted_position = position - self.p1
        # multiply row-wise shifted position and diffs
        dots = np.einsum('ij,ij->i', shifted_position, self.diffs)
        norm = np.linalg.norm(self.diffs, axis=1)
        coefs = dots / (norm + 1e-5)
        coefs = np.clip(coefs, 0, 1)
        projections = coefs[:, np.newaxis] * self.diffs 
        displacement_size = np.linalg.norm(projections - shifted_position, axis=1)
        closest_point = np.argmin(displacement_size)

        return np.sum(self.distances[:closest_point]) + np.linalg.norm(projections[closest_point])
    
    def closes_waypoint_distance(self):
        position = self._getDroneStateVector(0)[0:3]
        distances = np.linalg.norm(self.trajectory - position, axis=1)
        return np.min(distances)


    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        position = self._getDroneStateVector(0)

        reached_distance = self.get_travelled_distance()
        closest_waypoint_distance = self.closes_waypoint_distance()

        r_t = -10 if (abs(position[0]) > 1.5 or abs(position[1]) > 1.5 or position[2] > 2.0 # when the drone is too far away
            or abs(position[7]) > .4 or abs(position[8]) > .4 # when the drone is too tilted
        ) else 0
        r_p = reached_distance - self.reached_distance
        r_s = reached_distance
        r_wp = np.exp(-closest_waypoint_distance/self.dist_tol) if closest_waypoint_distance <= self.dist_tol else 0
        # TODO high body rates punishment

        k_p = 5
        k_wp = 5 
        k_s = 0.5
        # print(np.round(r_p, 2), np.round(r_wp, 2), np.round(r_s, 2))
        r = r_t + k_p * r_p + k_wp * r_wp + k_s * r_s
        self.reached_distance = reached_distance
        return r




    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.trajectory[-1] - state[0:3]) < .05:
            self.reset_vars()
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
            self.reset_vars()
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            self.reset_vars()
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
        return {"distance": self.reached_distance}

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
            # Position [0:3]
            # Orientation [3:7]
            # Roll, Pitch, Yaw [7:10]
            # Linear Velocity [10:13]
            # Angular Velocity [13:16]
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo]])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi]])
            #Add action buffer to observation space
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL, ActionType.ATTITUDE_PID]:
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

    def step(self,action):
        # # visualise trajectory - this is cheating, but it works
        if self.GUI and not self.visualised:
            self.visualised = True
            for point in self.trajectory:
                sphere_visual = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.03,
                    rgbaColor=[0, 1, 0, 1],
                    physicsClientId=self.CLIENT
                )
                target = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=sphere_visual,
                    basePosition=point,
                    useMaximalCoordinates=False,
                    physicsClientId=self.CLIENT
                )
                p.changeVisualShape(
                    target,
                    -1,
                    rgbaColor=[0.9, 0.3, 0.3, 1],
                    physicsClientId=self.CLIENT
                )
        return super().step(action)

    def update_waypoints(self):
        drone_position = self._getDroneStateVector(0)[0:3]
        current_waypoint = self.trajectory[self.current_waypoint_idx]
        if np.linalg.norm(drone_position - current_waypoint) < 0.1:
            self.current_waypoint_idx += 1
        


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
            ret = np.hstack([ret, self.trajectory[self.current_waypoint_idx:self.current_waypoint_idx+self.WAYPOINT_BUFFER_SIZE].reshape(1,-1)])
            return ret
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
