from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

import numpy as np
import copy
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from trajectories import TrajectoryFactory, DiscretizedTrajectory, Waypoint

class Rewards:

    def __init__(self, trajectory: np.ndarray) -> None:
        self.trajectory = trajectory
        self.p1 = self.trajectory[:-1]
        self.p2 = self.trajectory[1:]
        self.diffs = self.p2 - self.p1

    def forward(self, position: np.ndarray):
        return 0
    

    def __str__(self) -> str:
        return ""
    
txt_colour = [0,0,0]
txt_size = 2
txt_position = [0, 0, 0]

dummy_text = lambda txt, client_id: p.addUserDebugText(txt, 
                           txt_position,
                           lifeTime=0,
                           textSize=txt_size,
                           textColorRGB=txt_colour,
                           physicsClientId=client_id)

refreshed_text = lambda txt, client_id, replace_id: p.addUserDebugText(txt, 
                           txt_position,
                           lifeTime=0,
                           textSize=txt_size,
                           textColorRGB=txt_colour,
                           physicsClientId=client_id,
                           replaceItemUniqueId=replace_id)


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
        
        self.current_projection = np.array([0,0,0])
        self.current_projection_idx = 0

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
        drone = self._getDroneStateVector(0)[:3]
        self.projection_id = p.addUserDebugLine(drone, drone, [1,0,0], physicsClientId=self.CLIENT)
        self.text_id = dummy_text("Rewards: None", self.CLIENT)

    def reset_vars(self):
        self.current_waypoint_idx = 0
        self.reached_distance = 0
        self.current_projection = self.trajectory[0]
        self.current_projection_idx = 0

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
        norm = np.linalg.norm(self.diffs, axis=1) ** 2
        coefs = dots / (norm + 1e-5)
        coefs = np.clip(coefs, 0, 1)
        projections = coefs[:, np.newaxis] * self.diffs + self.p1
        displacement_size = np.linalg.norm(projections- position, axis=1)
        
        closest_point = np.argmin(displacement_size)

        self.current_projection = projections[closest_point]
        self.current_projection_idx = min(closest_point + 1, len(self.trajectory) - 1)

        return np.sum(self.distances[:closest_point]) + np.linalg.norm(projections[closest_point])
    
    def closest_waypoint_distance(self):
        position = self._getDroneStateVector(0)[0:3]
        distances = np.linalg.norm(self.trajectory - position, axis=1)
        return np.min(distances)


    def _computeReward(self):
        position = self._getDroneStateVector(0)

        reached_distance = self.get_travelled_distance()
        closest_waypoint_distance = self.closest_waypoint_distance()

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
        return r if closest_waypoint_distance < 0.2 else r_t




    ################################################################################

    def _computeTerminated(self):
        return False
        
    ################################################################################
    
    def _computeTruncated(self):
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
        return {"distance": self.reached_distance}

    def _observationSpace(self):
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

            # Add future waypoints to observation space
            obs_lower_bound = np.hstack([obs_lower_bound, np.array([[lo,lo,lo] for i in range(2)]).reshape(1, -1)])
            obs_upper_bound = np.hstack([obs_upper_bound, np.array([[hi,hi,hi] for i in range(2)]).reshape(1, -1)])

            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def step(self,action):
        # # visualise trajectory - this is cheating, but it works
        if self.GUI and not self.visualised:
            drone = self._getDroneStateVector(0)[:3]
            self.projection_id = p.addUserDebugLine(drone, drone, [1,0,0], physicsClientId=self.CLIENT)
            self.text_id = dummy_text("Rewards: None", self.CLIENT)

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
        else:
            self.projection_id = p.addUserDebugLine(self._getDroneStateVector(0)[0:3], self.current_projection, [1,0,0], physicsClientId=self.CLIENT, replaceItemUniqueId=self.projection_id)
            self.text_id = refreshed_text("Reward A: x; Reward B: y", self.CLIENT, self.text_id)
        
        return super().step(action)


    def _computeObs(self):
        """Returns the current observation of the environment.
        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.
        """

        if self.GUI:
            self.get_travelled_distance()
            print(self.current_projection)
            print(self.trajectory[self.current_projection_idx: self.current_projection_idx+2])

        #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
        obs = self._getDroneStateVector(0)
        ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(1, -1).astype('float32')
        #### Add action buffer to observation #######################
        # for i in range(self.ACTION_BUFFER_SIZE):
        #     ret = np.hstack([ret, np.array(self.action_buffer[i])])
        
        #### Add future waypoints to observation
        ret = np.hstack([ret, self.trajectory[self.current_projection_idx: self.current_projection_idx+2].reshape(1, -1)])
        return ret