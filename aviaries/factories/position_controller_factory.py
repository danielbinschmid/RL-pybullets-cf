import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from aviaries.PositionControllerAviary import PositionControllerAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import DiscretizedTrajectory
from stable_baselines3.common.vec_env import VecEnv
from agents.utils.configuration import Configuration
from .base_factory import BaseFactory
from trajectories.random_trajectory import RandomTrajectory
from trajectories.cube_trajectory import CubeTrajectory
from trajectories.trajectory import Trajectory


class PositionControllerFactory(BaseFactory):
    action_type: ActionType
    observation_type: ObservationType
    t_traj: DiscretizedTrajectory
    n_env_training: int
    initial_xyzs: np.ndarray
    seed: int 
    use_gui_for_test_env: bool
    output_path_location: str

    def __init__(self,
                 config: Configuration,
                 observation_type: ObservationType,
                 output_folder: str,
                 use_gui_for_test_env: bool = True,
                 n_env_training: int = 20,
                 seed: int = 0,
                 zero_velocity_at_target: bool = False,
                 training_trajectory: Trajectory = RandomTrajectory(),
                 testing_trajectory: Trajectory = CubeTrajectory()
        ) -> None:
        super().__init__()
        initial_xyzs = config.initial_xyzs
        action_type = config.action_type
        t_traj = config.t_traj

        self.initial_xyzs = initial_xyzs
        self.observation_type = observation_type
        self.action_type = action_type
        self.t_traj = t_traj
        self.n_env_training = n_env_training
        self.seed = seed
        self.use_gui_for_test_env = use_gui_for_test_env
        self.zero_velocity_at_target = zero_velocity_at_target,
        self.training_trajectory: Trajectory = training_trajectory
        self.testing_trajectory: Trajectory = testing_trajectory

    def get_train_env(self) -> VecEnv:
        train_env = make_vec_env(
            PositionControllerAviary,
            env_kwargs=dict(
                initial_xyzs=self.initial_xyzs,
                obs=self.observation_type,
                act=self.action_type,
                zero_velocity_at_target=self.zero_velocity_at_target,
                trajectory=self.training_trajectory
            ),
            n_envs=self.n_env_training,
            seed=self.seed
        )
        return train_env

    def get_eval_env(self):
        eval_env = PositionControllerAviary(
            initial_xyzs=self.initial_xyzs,
            obs=self.observation_type,
            act=self.action_type,
            zero_velocity_at_target=self.zero_velocity_at_target,
            trajectory=self.training_trajectory
        )
        return eval_env

    def get_test_env_gui(self):
        test_env = PositionControllerAviary(
            initial_xyzs=self.initial_xyzs,
            gui=self.use_gui_for_test_env,
            obs=self.observation_type,
            act=self.action_type,
            zero_velocity_at_target=self.zero_velocity_at_target,
            record=False,
            trajectory=self.testing_trajectory
        )
        return test_env

    def get_test_env_no_gui(self):
        test_env_nogui = PositionControllerAviary(
            initial_xyzs=self.initial_xyzs,
            obs=self.observation_type,
            act=self.action_type,
            zero_velocity_at_target=self.zero_velocity_at_target,
            trajectory=self.testing_trajectory
        )
        return test_env_nogui
