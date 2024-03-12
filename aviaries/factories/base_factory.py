import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from aviaries.PositionControllerAviary import PositionControllerAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import DiscretizedTrajectory
from stable_baselines3.common.vec_env import VecEnv
from aviaries.configuration import Configuration
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary


class BaseFactory:

    def __init__(self) -> None:
        pass

    def get_train_env(self) -> VecEnv:
        raise NotImplementedError("Must be overwritten by child class.")

    def get_eval_env(self) -> BaseRLAviary:
        raise NotImplementedError("Must be overwritten by child class.")

    def get_test_env_gui(self) -> BaseRLAviary:
        raise NotImplementedError("Must be overwritten by child class.")

    def get_test_env_no_gui(self) -> BaseRLAviary:
        raise NotImplementedError("Must be overwritten by child class.")
