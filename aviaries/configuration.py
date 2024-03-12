import numpy as np
from gym_pybullet_drones.utils.enums import ActionType
from trajectories import DiscretizedTrajectory
import numpy as np
from typing import List


class Configuration:
    action_type: ActionType
    initial_xyzs: np.ndarray
    t_traj: DiscretizedTrajectory
    target_reward: float
    output_path_location: str
    n_timesteps: int
    local: bool
    episode_len_sec: float
    waypoint_buffer_size: int
    k_p: float
    k_wp: float
    k_s: float
    max_reward_distance: float
    waypoint_dist_tol: float

    def __init__(
        self,
        action_type: ActionType,
        initial_xyzs: np.ndarray,
        output_path_location: str,
        n_timesteps: int,
        local: bool,
        episode_len_sec: int = 10,
        waypoint_buffer_size: int = 2,
        k_p: float = 1,
        k_wp: float = 1,
        k_s: float = 1,
        max_reward_distance: float = 1,
        waypoint_dist_tol: float = 1,
        t_traj=None,
        t_reward=None,
    ) -> None:
        self.action_type = action_type
        self.initial_xyzs = initial_xyzs
        self.t_traj = t_traj

        if t_reward is None:
            self.target_reward = float("inf")
        else:
            t_reward = t_reward
        self.output_path_location = output_path_location
        self.n_timesteps = n_timesteps
        self.local = local
        self.episode_len_sec = episode_len_sec
        self.waypoint_buffer_size = waypoint_buffer_size
        self.k_p = k_p
        self.k_wp = k_wp
        self.k_s = k_s
        self.max_reward_distance = max_reward_distance
        self.waypoint_dist_tol = waypoint_dist_tol

    def update_trajectory(self, t_traj: DiscretizedTrajectory, initial_xyz):
        self.t_traj = t_traj
        self.initial_xyzs = initial_xyz
