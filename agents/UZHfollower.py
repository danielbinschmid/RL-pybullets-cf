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

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False
DEFAULT_TIMESTEPS = 2e4

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO, local=True,
        timesteps=DEFAULT_TIMESTEPS):

    # CONFIG ##################################################

    # target trajectory
    t_wps = TrajectoryFactory.waypoints_from_numpy(
        np.asarray([
            [0, 0, 0],
            [0, 0, 0.2],
            [0, 0, 0.4],
            [0, 0, 0.6],
            [0, 0, 0.8],
            [0, 0, 1],
            [0, 0.2, 1],
            [0, 0.4, 1],
            [0, 0.6, 1],
            [0, 0.8, 1],
            [0, 1, 1],
            [0.2, 1, 1],
            [0.4, 1, 1],
            [0.6, 1, 1],
            [0.8, 1, 1],
            [1, 1, 1],
        ])
    )
    initial_xyzs = np.array([[0.,     0.,     0.]])
    t_traj = TrajectoryFactory.get_discr_from_wps(t_wps)

    # output path location
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # target reward 
    target_reward = 3e3

    # #########################################################
    
    # ENVS ####################################################
        
    if multiagent:
        raise NotImplementedError("not implemented")
    
    train_env = make_vec_env(
        UZHAviary,
        env_kwargs=dict(
            target_trajectory=t_traj,
            initial_xyzs=initial_xyzs,
            obs=DEFAULT_OBS, 
            act=DEFAULT_ACT
        ),
        n_envs=20,
        seed=0
    )
    eval_env = UZHAviary(
        target_trajectory=t_traj,
        initial_xyzs=initial_xyzs,
        obs=DEFAULT_OBS, 
        act=DEFAULT_ACT
    )

    # #########################################################

    # SETUP ###################################################

    # model
    model = PPO('MlpPolicy',
                train_env,
                # tensorboard_log=filename+'/tb/',
                verbose=1)

    # callbacks
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward,
        verbose=1
    )
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    
    # ##########################################################

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)
    print('[INFO] Number of timesteps:', timesteps)

    # TRAIN ####################################################

    # fit
    model.learn(total_timesteps=timesteps,
                callback=eval_callback,
                log_interval=100)
    
    # save model
    model.save(filename+'/final_model.zip')
    print(filename)

    # print training progression 
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))
    
    # ##########################################################

    # TEST #####################################################

    test_env = UZHAviary(
        target_trajectory=t_traj,
        initial_xyzs=initial_xyzs,
        gui=gui,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        record=record_video
    )
    test_env_nogui = UZHAviary(
        target_trajectory=t_traj,
        initial_xyzs=initial_xyzs,
        obs=DEFAULT_OBS, 
        act=DEFAULT_ACT
    )
    test_simple_follower(
        local=local,
        filename=filename,
        test_env_nogui=test_env_nogui,
        test_env=test_env,
        output_folder=output_folder
    )

    # ##########################################################


    

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--timesteps',          default=DEFAULT_TIMESTEPS,     type=int,           help='number of train timesteps before stopping', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
