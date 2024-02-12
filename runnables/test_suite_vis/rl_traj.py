import argparse
import numpy as np
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import TrajectoryFactory
from agents.utils.configuration import Configuration
from aviaries.factories.uzh_trajectory_follower_factory import TrajectoryFollowerAviaryFactory

from agents.test_policy import run_test
from agents.train_policy import run_train

from torch import nn

###### INFRASTRUCTURE PARAMS #######
GUI = True
RECORD_VIDEO = False
OUTPUT_FOLDER = 'results'
COLAB = False
####################################

###### USUALLY NOT CHANGED #########
OBS = ObservationType('kin') # 'kin' or 'rgb'
ACT = ActionType.ATTITUDE_PID
AGENTS = 1
NUM_DRONES = 1
CTRL_FREQ = 10
MA = False
####################################

###### TEST TRAIN FLAGS ############
TRAIN = False
TEST = True
####################################

###### ENVIRONMENT PARAMS ##########
TIMESTEPS = 2.5e6
N_ENVS = 20
EPISODE_LEN_SEC = 15
####################################

###### HYPERPARAMS #################
WAYPOINT_BUFFER_SIZE = 3
K_P = 0.05
K_WP = 50
K_S = 0.02
MAX_REWARD_DISTANCE = 0.03
WAYPOINT_DIST_TOL = 0.2
####################################


def init_targets():

    base_traj = TrajectoryFactory.get_linear_square_traj_discretized(
        n_discretization_level=4
    )
    initial_xyzs = base_traj[0].coordinate.reshape((1, 3))
    wps = [wp for wp in base_traj]
    t_traj = TrajectoryFactory.get_pol_discretized_trajectory(
        t_waypoints=wps,
        n_points_discretization_level=20
    )
    return t_traj, initial_xyzs

def run(output_folder=OUTPUT_FOLDER,
        gui=GUI,
        timesteps=TIMESTEPS,
        train: bool = TRAIN,
        test: bool = TEST,
        n_envs: int = N_ENVS,
        episode_len_sec: int = EPISODE_LEN_SEC,
        waypoint_buffer_size: int = WAYPOINT_BUFFER_SIZE,
        k_p: float = K_P,
        k_wp: float = K_WP,
        k_s: float = K_S,
        max_reward_distance: float = MAX_REWARD_DISTANCE,
        waypoint_dist_tol: float = WAYPOINT_DIST_TOL,
    ):


    # CONFIG ##################################################
    t_traj, init_wp = init_targets()

    # random number in range 10-99
    if train:
        output_folder = f"{output_folder}/k_p={k_p}_k_wp={k_wp}_k_s={k_s}_max_reward_distance={max_reward_distance}_waypoint_dist_tol={waypoint_dist_tol}"
    print(f"Output folder: {output_folder}")

    config = Configuration(
        action_type=ACT,
        initial_xyzs=init_wp,
        output_path_location=output_folder,
        n_timesteps=timesteps,
        t_traj=t_traj,
        local=True,
        episode_len_sec=episode_len_sec,
        waypoint_buffer_size=waypoint_buffer_size,
        k_p=k_p,
        k_wp=k_wp,
        k_s=k_s,
        max_reward_distance=max_reward_distance,
        waypoint_dist_tol=waypoint_dist_tol
    )
    
    env_factory = TrajectoryFollowerAviaryFactory(
        config=config,
        observation_type=OBS,
        use_gui_for_test_env=gui,
        n_env_training=n_envs,
        seed=0,
        single_traj=True
    )

    # if train:
    #     run_train(config=config,
    #               env_factory=env_factory)

    if test:
        run_test(config=config,
                env_factory=env_factory)


    

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                    default=GUI,                    type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--output_folder',          default=OUTPUT_FOLDER,          type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--timesteps',              default=TIMESTEPS,              type=int,           help='number of train timesteps before stopping', metavar='')
    parser.add_argument('--train',                  default=TRAIN,                  type=str2bool,      help='Whether to train (default: True)', metavar='')
    parser.add_argument('--test',                   default=TEST,                   type=str2bool,      help='Whether to test (default: True)', metavar='')
    parser.add_argument('--n_envs',                 default=N_ENVS,                 type=int,           help='number of parallel environments', metavar='')
    parser.add_argument('--episode_len_sec',        default=EPISODE_LEN_SEC,        type=int,           help='number of parallel environments', metavar='')
    parser.add_argument('--waypoint_buffer_size',   default=WAYPOINT_BUFFER_SIZE,   type=int,           help='number of parallel environments', metavar='')
    parser.add_argument('--k_p',                    default=K_P,                    type=float,         help='number of parallel environments', metavar='')
    parser.add_argument('--k_wp',                   default=K_WP,                   type=float,         help='number of parallel environments', metavar='')
    parser.add_argument('--k_s',                    default=K_S,                    type=float,         help='number of parallel environments', metavar='')
    parser.add_argument('--max_reward_distance',    default=MAX_REWARD_DISTANCE,    type=float,         help='number of parallel environments', metavar='')
    parser.add_argument('--waypoint_dist_tol',      default=WAYPOINT_DIST_TOL,      type=float,         help='number of parallel environments', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))