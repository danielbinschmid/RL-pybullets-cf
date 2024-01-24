"""Script learns an agent to follow a target trajectory.
"""

import os
from datetime import datetime
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from aviaries.SimpleFollowerAviary import SimpleFollowerAviary
from aviaries.FollowerAviary import FollowerAviary
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import TrajectoryFactory, Waypoint, DiscretizedTrajectory
from agents.test_simple_follower import test_simple_follower

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType.ONE_D_RPM # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False
DEFAULT_TIMESTEPS = 3e5

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO, local=True,
        timesteps=DEFAULT_TIMESTEPS,
        action_type: str='rpm'):

    # CONFIG ##################################################
    if action_type == 'rpm':
        action_type = ActionType.RPM
    elif action_type == 'one_d_rpm':
        action_type = ActionType.ONE_D_RPM
    elif action_type == 'attitude':
        action_type = ActionType.ATTITUDE_PID
    else:
        raise ValueError(f'Specified not implemented action type {action_type}.')
    
    # target trajectory
    t_wps = TrajectoryFactory.waypoints_from_numpy(
        np.asarray([
            [0, 0.5, 0.5],
        ])
    )
    initial_xyzs = np.array([[0.,     0.,     0.1]])
    t_traj = TrajectoryFactory.get_discr_from_wps(t_wps)

    filename = "/shared/d/master/ws23/UAV-lab/git_repos/RL-pybullets-cf/agents/rpm_task_tilted/save-01.24.2024_19.05.55/best_model.zip"
    
    # ##########################################################

    # TEST #####################################################

    test_env = SimpleFollowerAviary(
        target_trajectory=t_traj,
        initial_xyzs=initial_xyzs,
        gui=gui,
        obs=DEFAULT_OBS,
        act=action_type,
        record=record_video
    )
    test_env_nogui = SimpleFollowerAviary(
        target_trajectory=t_traj,
        initial_xyzs=initial_xyzs,
        obs=DEFAULT_OBS, 
        act=action_type
    )

    if local and gui:
        input("Press Enter to continue...")

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
    parser.add_argument('--action_type',          default=DEFAULT_TIMESTEPS,     type=str,           help='Either "one_d_rpm", "rpm" or "attitude"', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
