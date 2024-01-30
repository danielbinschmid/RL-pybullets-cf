"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from aviaries.UZHAviary import UZHAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import TrajectoryFactory, Waypoint, DiscretizedTrajectory
from agents.test_policy import test_simple_follower
from agents.utils.configuration import Configuration
from factories.uzh_trajectory_follower_factory import TrajectoryFollowerAviaryFactory

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType.ATTITUDE_PID
DEFAULT_AGENTS = 1
DEFAULT_MA = False
DEFAULT_TIMESTEPS = 1e5
DEFAULT_N_ENVS = 5
DEFAULT_TRAIN = True
DEFAULT_TEST = True
from agents.test_policy import run_test
from agents.train_policy import run_train

def init_targets():
    t_wps = TrajectoryFactory.waypoints_from_numpy(
        np.asarray([
            [0, 0, 0.2],
            [0, 0, 0.4],
            [0, 0, 0.6],
            [0, 0, 0.8],
            [0, 0, 1],
        ])
    )
    initial_xyzs = np.array([[0.,     0.,     0.]])
    t_traj = TrajectoryFactory.get_discr_from_wps(t_wps)
    return t_traj, initial_xyzs

def run(output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI, n_envs = DEFAULT_N_ENVS,
        timesteps=DEFAULT_TIMESTEPS,
        train: bool = DEFAULT_TRAIN,
        test: bool = DEFAULT_TEST):

    # CONFIG ##################################################
    t_traj, init_wp = init_targets()

    config = Configuration(
        action_type=DEFAULT_ACT,
        initial_xyzs=init_wp,
        output_path_location=output_folder,
        n_timesteps=timesteps,
        t_traj=t_traj,
        local=True
    )
    
    env_factory = TrajectoryFollowerAviaryFactory(
        config=config,
        observation_type=DEFAULT_OBS,
        use_gui_for_test_env=gui,
        n_env_training=n_envs,
        seed=0
    )

    if train:
        run_train(config=config,
                  env_factory=env_factory)

    if test:
        run_test(config=config,
                 env_factory=env_factory)


    

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--timesteps',          default=DEFAULT_TIMESTEPS,     type=int,           help='number of train timesteps before stopping', metavar='')
    parser.add_argument('--train',          default=DEFAULT_TRAIN,     type=str2bool,           help='Whether to train (default: True)', metavar='')
    parser.add_argument('--test',          default=DEFAULT_TEST,     type=str2bool,           help='Whether to test (default: True)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
