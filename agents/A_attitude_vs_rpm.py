"""
RUNS EXPERIMENT A.

3 MODES: 
- DOWN (from [0,0,1] to [0,0,0.1])
- UP (from [0,0,0.1] to [0,0,1])
- SIDEWAYS (from [0,0,1] to [0,1,1])
- DIAGONAL_UP (from [0,0,0.1] to [0,1,1])
- DIAGONAL_DOWN (from [0,0,1] to [0,1,0.1])

- every mode 10 times

"""
import argparse
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType
from aviaries.factories.simple_follower_factory import EnvFactorySimpleFollowerAviary
from agents.utils.configuration import Configuration
from train_policy import run_train
from test_policy import run_test
from agents.utils.configuration import Configuration
from gym_pybullet_drones.utils.enums import ActionType
from typing import List
from trajectories import TrajectoryFactory
import numpy as np
import os

# defaults for command line arguments
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_GUI = True
DEFAULT_TIMESTEPS = 1e6
DEFAULT_ACTION_TYPE = 'rpm' # 'rpm', 'one_d_rpm', 'attitude'
DEFAULT_TRAIN = True 
DEFAULT_TEST = True
DEFAULT_MODE = "UP" # DOWN, UP, SIDEWAYS, DIAGONAL_UP, DIAGONAL_DOWN

# more configurations
DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'


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

def parse_mode(mode:str):
    init_wp = None 
    t_wp = None

    if mode == "DOWN":
        init_wp     = [0,0,1]
        t_wp        = [0,0,0.1]
    elif mode == "UP":
        init_wp     = [0,0,0.1]
        t_wp        = [0,0,1]
    elif mode == "SIDEWAYS":
        init_wp     = [0,0,1]
        t_wp        = [0,1,1]
    elif mode == "DIAGONAL_UP":
        init_wp     = [0,0,0.1]
        t_wp        = [0,1,1]
    elif mode == "DIAGONAL_DOWN":
        init_wp     = [0,0,1]
        t_wp        = [0,1,0.1]
    else:
        raise ValueError(f'Invalide mode {mode}')

    return t_wp, init_wp


def run(output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        timesteps=DEFAULT_TIMESTEPS,
        action_type: str='rpm',
        train: bool=DEFAULT_TRAIN,
        test: bool=DEFAULT_TEST,
        mode: str = DEFAULT_MODE
    ):

    t_wp, init_wp = parse_mode(mode)
    print("output folder ", output_folder)
    config: Configuration = parse_config(
        t_waypoint=t_wp,
        initial_waypoint=init_wp,
        action_type=action_type,
        output_folder=output_folder,
        n_timesteps=timesteps,
        local=False
    )

    env_factory = EnvFactorySimpleFollowerAviary(
        config=config,
        output_folder=output_folder,
        observation_type=DEFAULT_OBS,
        use_gui_for_test_env=gui,
        n_env_training=20,
        seed=0,
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
    parser.add_argument('--action_type',          default=DEFAULT_TIMESTEPS,     type=str,           help='Either "one_d_rpm", "rpm" or "attitude"', metavar='')
    parser.add_argument('--train',          default=DEFAULT_TRAIN,     type=str2bool,           help='Whether to train (default: True)', metavar='')
    parser.add_argument('--test',          default=DEFAULT_TEST,     type=str2bool,           help='Whether to test (default: True)', metavar='')
    parser.add_argument('--mode',          default=DEFAULT_MODE,     type=str,           help='Experiment mode (default "UP")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
