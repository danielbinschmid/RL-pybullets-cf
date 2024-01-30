from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

import numpy as np
import copy
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from trajectories import TrajectoryFactory, DiscretizedTrajectory, Waypoint

class RewardDict: 
    def __init__(self, r_t: float=0, r_p: float=0, r_wp:float=0, r_s: float=0) -> None:
        self.r_t = r_t
        self.r_p = r_p 
        self.r_wp = r_wp 
        self.r_s = r_s 
    
    def __str__(self) -> str:
        return f'r_t: {self.r_t}; r_p: {self.r_p}; r_wp: {self.r_wp}; r_s: {self.r_s}'

    def sum(self):
        return self.r_t + self.r_p + self.r_wp + self.r_s
    
class Rewards:
    cur_reward: RewardDict

    def __init__(self, trajectory: np.ndarray, k_p: float=5, k_wp: float=5, k_s: float=0.5) -> None:
        self.trajectory = trajectory

        # intermediates
        self.p1 = self.trajectory[:-1]
        self.p2 = self.trajectory[1:]
        self.diffs = self.p2 - self.p1
        self.distances = np.linalg.norm(self.p1 - self.p2, axis=1)
        self.reached_distance = 0

        # weights for reward
        self.k_p = k_p
        self.k_wp = k_wp
        self.k_s = k_s 
        
        self.dist_tol = 0.08
        self.cur_reward = RewardDict()

    def get_projections(self, position: np.ndarray):
        shifted_position = position - self.p1
        dots = np.einsum('ij,ij->i', shifted_position, self.diffs)
        norm = np.linalg.norm(self.diffs, axis=1) ** 2
        coefs = dots / (norm + 1e-5)
        coefs = np.clip(coefs, 0, 1)
        projections = coefs[:, np.newaxis] * self.diffs + self.p1
        return projections
    
    def get_travelled_distance(self, position: np.ndarray):
        projections = self.get_projections(position)
        displacement_size = np.linalg.norm(projections- position, axis=1)
        closest_point_idx = np.argmin(displacement_size)

        current_projection = projections[closest_point_idx]
        current_projection_idx = min(closest_point_idx + 1, len(self.trajectory) - 1)

        overall_distance_travelled = np.sum(self.distances[:closest_point_idx]) \
            + np.linalg.norm(projections[closest_point_idx] - self.p1[closest_point_idx])

        return current_projection, current_projection_idx, overall_distance_travelled
    
    def closest_waypoint_distance(self, position: np.ndarray):
        distances = np.linalg.norm(self.trajectory - position, axis=1)
        return np.min(distances)

    def weight_rewards(self, r_t, r_p, r_wp, r_s):
        self.cur_reward = RewardDict(
            r_t=r_t,
            r_p=self.k_p * r_p,
            r_wp=self.k_wp * r_wp,
            r_s=self.k_s * r_s
        )

        return self.cur_reward.sum()

    def compute_reward(self, drone_state: np.ndarray, reached_distance: np.ndarray):
        """
        TODO high body rates punishment
        """
        position = drone_state[:3]
        closest_waypoint_distance = self.closest_waypoint_distance(position)

        r_t = -10 if (abs(position[0]) > 1.5 or abs(position[1]) > 1.5 or position[2] > 2.0 # when the drone is too far away
            or abs(drone_state[7]) > .4 or abs(drone_state[8]) > .4 # when the drone is too tilted
        ) else 0
        r_p = reached_distance - self.reached_distance
        r_s = reached_distance
        r_wp = np.exp(-closest_waypoint_distance/self.dist_tol) if closest_waypoint_distance <= self.dist_tol else 0


        r = self.weight_rewards(r_t, r_p, r_wp, r_s)
        self.reached_distance = reached_distance

        return r if closest_waypoint_distance < 0.2 else r_t
    
    def __str__(self) -> str:
        return ""