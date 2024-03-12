import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from aviaries.PositionControllerAviaryTest import PositionControllerAviaryTest
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import DiscretizedTrajectory
from stable_baselines3.common.vec_env import VecEnv
from aviaries.configuration import Configuration
from .base_factory import BaseFactory


class PositionControllerFactoryTest(BaseFactory):
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
                 n_env_training: int=20,
                 seed: int = 0,
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

    def get_train_env(self) -> VecEnv:
        train_env = make_vec_env(
            PositionControllerAviaryTest,
            env_kwargs=dict(
                target_trajectory=self.t_traj,
                initial_xyzs=self.initial_xyzs,
                obs=self.observation_type,
                act=self.action_type
            ),
            n_envs=self.n_env_training,
            seed=self.seed
        )
        return train_env

    def get_eval_env(self):
        eval_env = PositionControllerAviaryTest(
            target_trajectory=self.t_traj,
            initial_xyzs=self.initial_xyzs,
            obs=self.observation_type,
            act=self.action_type
        )
        return eval_env

    def get_test_env_gui(self):
        test_env = PositionControllerAviaryTest(
            target_trajectory=self.t_traj,
            initial_xyzs=self.initial_xyzs,
            gui=self.use_gui_for_test_env,
            obs=self.observation_type,
            act=self.action_type,
            record=False
        )
        return test_env

    def get_test_env_no_gui(self):
        test_env_nogui = PositionControllerAviaryTest(
            target_trajectory=self.t_traj,
            initial_xyzs=self.initial_xyzs,
            obs=self.observation_type,
            act=self.action_type
        )
        return test_env_nogui
