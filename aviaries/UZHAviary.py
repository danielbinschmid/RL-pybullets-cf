from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

import numpy as np
import pybullet as p
from gymnasium import spaces

from trajectories import TrajectoryFactory, DiscretizedTrajectory
from aviaries.rewards.uzh_trajectory_reward import Rewards
    
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
                 pyb_freq: int = 120,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 episode_len_sec: int = 8,
                 waypoint_buffer_size: int = 2,
                 k_p: float = 1.0,
                 k_wp: float = 1.0,
                 k_s: float = 0.0,
                 max_reward_distance: float = 0.2,
                 waypoint_dist_tol: float = 0.12
                 ):
        
        self.EPISODE_LEN_SEC = episode_len_sec
        self.NUM_DRONES = 1
        self.INIT_XYZS = initial_xyzs


        # FOR DEVELOPMENT 
        self.one_traj = False
        self.single_traj = target_trajectory

        # TRAJECTORY
        self.WAYPOINT_BUFFER_SIZE = waypoint_buffer_size # how many steps into future to interpolate
        self.trajectory = self.set_trajectory()
        assert self.WAYPOINT_BUFFER_SIZE < len(self.trajectory), "Buffer size should be smaller than the number of waypoints"
        self.current_waypoint_idx = 0
        self.current_projection_idx = 0
        self.furthest_reached_waypoint_idx = 0
        self.future_waypoints_relative = self.trajectory[self.current_projection_idx: self.current_projection_idx+self.WAYPOINT_BUFFER_SIZE] - self.trajectory[self.current_projection_idx]
        self.rewards = Rewards(
            trajectory=self.trajectory,
            k_p=k_p,
            k_wp=k_wp,
            k_s=k_s,
            max_reward_distance=max_reward_distance,
            dist_tol=waypoint_dist_tol
        )
        self.current_action = None
        self.INIT_XYZS = self.trajectory[0]

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

        
        # Visualisation
        self.current_projection = np.array([0,0,0])
        self.visualised = False
        drone_pos = self._getDroneStateVector(0)[:3]
        self.projection_id = p.addUserDebugLine(drone_pos, drone_pos, [0.3,0.3,0.3], physicsClientId=self.CLIENT)
        self.waypoint_connection_ids = [
            p.addUserDebugLine(drone_pos, drone_pos, [0,0,0], physicsClientId=self.CLIENT) for i in range(self.WAYPOINT_BUFFER_SIZE)
        ]
        self.text_id = dummy_text("Rewards: None", self.CLIENT)

    def reset_vars(self):
        self.current_waypoint_idx = 0
        self.rewards.reached_distance = 0
        self.current_projection = self.trajectory[0]
        self.current_projection_idx = 0
        self.self_trajectory = self.set_trajectory()
        self.rewards.reset(self.self_trajectory)

    def set_trajectory(self):
        if self.one_traj:
            trajectory = np.array([x.coordinate for x in self.single_traj])
        else:
            ctrl_wps = TrajectoryFactory.gen_random_trajectory(
                start=np.array([0, 0, 1]),
                n_discr_level=20,
                n_ctrl_points=10,
                std_dev_deg=30,
                distance_between_ctrl_points=1.3,
                init_dir=None,
                return_ctrl_points=False
            )
            trajectory = [x.coordinate for x in ctrl_wps]

        return np.vstack([
            trajectory,
            np.array(trajectory[-1] * np.ones((self.WAYPOINT_BUFFER_SIZE, 3)))
        ])

    def _computeReward(self):
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[:3]

        self.current_projection, self.current_projection_idx, reached_distance = self.rewards.get_travelled_distance(drone_pos)
        # self.furthest_reached_waypoint_idx = max(self.furthest_reached_waypoint_idx, self.current_projection_idx)
        self.furthest_reached_waypoint_idx = self.current_projection_idx

        r = self.rewards.compute_reward(
            drone_state=drone_state,
            reached_distance=reached_distance,
            bodyrates=self.current_action[0, 1:4]
        )
        return r

    def _computeTerminated(self):
        return False
        
    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[7]) > .4 or abs(state[8]) > .4):
            self.reset_vars()
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            self.reset_vars()
            return True
        else:
            return False

    def _computeInfo(self):
        return {"distance": self.rewards.reached_distance}

    def _observationSpace(self):
        # OBS SPACE OF SIZE 12
        # Observation vector - X Y Z Q1 Q2 Q3 Q4 R P Y VX VY VZ WX WY WZ
        # Position [0:3]
        # Orientation [3:7]
        # Roll, Pitch, Yaw [7:10]
        # Linear Velocity [10:13]
        # Angular Velocity [13:16]
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[0, lo,lo,lo,lo,lo,lo,lo,lo,lo]])
        obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi]])

        # Add future waypoints to observation space
        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[lo,lo,lo] for i in range(self.WAYPOINT_BUFFER_SIZE+1)]).reshape(1, -1)])
        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[hi,hi,hi] for i in range(self.WAYPOINT_BUFFER_SIZE+1)]).reshape(1, -1)])

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
    
    ################################################################################

    def step(self,action):
        self.current_action = action
        # # visualise trajectory - this is cheating, but it works
        if self.GUI and not self.visualised:
            drone = self._getDroneStateVector(0)[:3]
            self.projection_id = p.addUserDebugLine(drone, drone, [1,0,0], physicsClientId=self.CLIENT)
            for i, wp in enumerate(self.waypoint_connection_ids):
                p.addUserDebugLine(drone, drone, [0,0,0], physicsClientId=self.CLIENT, replaceItemUniqueId=wp)
            self.text_id = dummy_text("Rewards: None", self.CLIENT)

            self.visualised = True
            for point in self.trajectory:
                sphere_visual = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.03,
                    rgbaColor=[0, 1, 0, 0.6],
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
                    rgbaColor=[0.9, 0.3, 0.3, 0.6],
                    physicsClientId=self.CLIENT
                )
        else:
            drone_pos = self._getDroneStateVector(0)[:3]
            self.projection_id = p.addUserDebugLine(drone_pos, self.current_projection, [1,0,0], physicsClientId=self.CLIENT, replaceItemUniqueId=self.projection_id)
            # print(self.future_waypoints_relative)
            for i in range(len(self.waypoint_connection_ids)):
                self.waypoint_connection_ids[i] = \
                    p.addUserDebugLine(drone_pos, self.future_waypoints_relative[i], [0.3,0.3,0.3], physicsClientId=self.CLIENT, replaceItemUniqueId=self.waypoint_connection_ids[i])

            self.text_id = refreshed_text(str(self.rewards.cur_reward), self.CLIENT, self.text_id)
        
        return super().step(action)

    def _computeObs(self):
        obs = self._getDroneStateVector(0)
        ret = np.hstack([obs[2], obs[7:10], obs[10:13], obs[13:16]]).reshape(1, -1).astype('float32')
        self.future_waypoints_relative = self.trajectory[self.furthest_reached_waypoint_idx:self.furthest_reached_waypoint_idx+self.WAYPOINT_BUFFER_SIZE]

        #### Add relative positions of future waypoints to observation
        ret = np.hstack([ret, self.current_projection - obs[:3].reshape(1, -1).astype('float32'), (self.future_waypoints_relative - obs[:3]).reshape(1, -1).astype('float32')])
        return ret