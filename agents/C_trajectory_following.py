import argparse
import numpy as np
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import TrajectoryFactory
from agents.utils.configuration import Configuration
from factories.uzh_trajectory_follower_factory import TrajectoryFollowerAviaryFactory

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
CTRL_FREQ = 30
MA = False
####################################

###### TEST TRAIN FLAGS ############
TRAIN = True
TEST = True
####################################

###### ENVIRONMENT PARAMS ##########
TIMESTEPS = 1e6
N_ENVS = 5
EPISODE_LEN_SEC = 8
####################################

###### HYPERPARAMS #################
WAYPOINT_BUFFER_SIZE = 2
K_P = 1
K_WP = 3
K_S = 0.1
MAX_REWARD_DISTANCE = 0.2 
WAYPOINT_DIST_TOL = 0.12
####################################

from agents.test_policy import run_test
from agents.train_policy import run_train

def init_targets():
    points_per_segment = 4
    z_segment = np.array([
        [0, 0, (1/points_per_segment)*i] for i in range(1, points_per_segment + 1)
    ])
    y_segment = np.array([
        [0, (1/points_per_segment)*i, 1] for i in range(1, points_per_segment + 1)
    ])
    x_segment = np.array([
        [(1/points_per_segment)*i, 1, 1] for i in range(1, points_per_segment + 1)
    ])
    initial_xyzs = np.array([[0.,     0.,     0.]])
    pts = np.vstack([initial_xyzs, z_segment, y_segment, x_segment])
    t_wps = TrajectoryFactory.waypoints_from_numpy(
        pts
    )
    t_traj = TrajectoryFactory.get_discr_from_wps(t_wps)
    return t_traj, initial_xyzs

def run(output_folder=OUTPUT_FOLDER,
        gui=GUI, n_envs = N_ENVS,
        timesteps=TIMESTEPS,
        train: bool = TRAIN,
        test: bool = TEST):

    # CONFIG ##################################################
    t_traj, init_wp = init_targets()

    config = Configuration(
        action_type=ACT,
        initial_xyzs=init_wp,
        output_path_location=output_folder,
        n_timesteps=timesteps,
        t_traj=t_traj,
        local=True
    )
    
    env_factory = TrajectoryFollowerAviaryFactory(
        config=config,
        observation_type=OBS,
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
    parser.add_argument('--gui',                default=GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--output_folder',      default=OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--timesteps',          default=TIMESTEPS,     type=int,           help='number of train timesteps before stopping', metavar='')
    parser.add_argument('--train',          default=TRAIN,     type=str2bool,           help='Whether to train (default: True)', metavar='')
    parser.add_argument('--test',          default=TEST,     type=str2bool,           help='Whether to test (default: True)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
