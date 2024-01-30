import numpy as np
from gym_pybullet_drones.utils.enums import ActionType
from trajectories import DiscretizedTrajectory
import os
import numpy as np
from trajectories import TrajectoryFactory
from typing import List



class Configuration:
    action_type: ActionType
    initial_xyzs: np.ndarray
    t_traj: DiscretizedTrajectory
    target_reward: float
    output_path_location: str
    n_timesteps: int
    local: bool 

    def __init__(self, 
                 action_type: ActionType, 
                 initial_xyzs: np.ndarray, output_path_location: str, n_timesteps: int, local: bool,  t_traj=None, t_reward = None) -> None:
        self.action_type = action_type
        self.initial_xyzs = initial_xyzs
        self.t_traj = t_traj

        if t_reward is None:
            self.target_reward = float('inf')
        else:
            t_reward = t_reward
        self.output_path_location = output_path_location
        self.n_timesteps = n_timesteps
        self.local = local


