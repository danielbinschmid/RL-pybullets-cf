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

    def __init__(self, action_type: ActionType, initial_xyzs: np.ndarray, t_traj: DiscretizedTrajectory, output_path_location: str, n_timesteps: int, local: bool) -> None:
        self.action_type = action_type
        self.initial_xyzs = initial_xyzs
        self.t_traj = t_traj
        self.target_reward = float('inf')
        self.output_path_location = output_path_location
        self.n_timesteps = n_timesteps
        self.local = local

    
def parse_config(
        t_waypoint: List[float], 
        initial_waypoint: List[float], 
        action_type: str, 
        output_folder: str, 
        n_timesteps: int, 
        local: bool) -> Configuration:

    # parse action type
    action_type_parsed = None
    if action_type == 'rpm':
        action_type_parsed = ActionType.RPM
    elif action_type == 'one_d_rpm':
        action_type_parsed = ActionType.ONE_D_RPM
    elif action_type == 'attitude':
        action_type_parsed = ActionType.ATTITUDE_PID
    else:
        raise ValueError(f'Specified not implemented action type {action_type}.')     

    # target trajectory and initial point
    t_wps = TrajectoryFactory.waypoints_from_numpy(
        np.asarray([
            t_waypoint,
        ])
    )
    initial_xyzs = np.array([initial_waypoint])
    t_traj = TrajectoryFactory.get_discr_from_wps(t_wps)

    # output path location
    if not os.path.exists(output_folder):
        os.makedirs(output_folder+'/')

    config = Configuration(
        action_type=action_type_parsed,
        initial_xyzs=initial_xyzs,
        t_traj=t_traj,
        output_path_location=output_folder,
        n_timesteps=n_timesteps,
        local=local
    )

    return config