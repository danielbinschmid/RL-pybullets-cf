import sys 
sys.path.append("../..")
import argparse
import numpy as np
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import TrajectoryFactory, DiscretizedTrajectory
from agents.utils.configuration import Configuration
from aviaries.factories.uzh_trajectory_follower_factory import TrajectoryFollowerAviaryFactory

from agents.test_policy import run_test
from agents.train_policy import run_train
from runnables.test_suite_eval.eval_tracks import load_eval_tracks 
from torch import nn
from typing import List, Dict
from tqdm import tqdm
import json 

###### INFRASTRUCTURE PARAMS #######
GUI = True
RECORD_VIDEO = False
OUTPUT_FOLDER = 'checkpointed_models'
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
EPISODE_LEN_SEC = 50
####################################

###### HYPERPARAMS #################
WAYPOINT_BUFFER_SIZE = 2
K_P = 5
K_WP = 8
K_S = 0.05
MAX_REWARD_DISTANCE = 0.0
WAYPOINT_DIST_TOL = 0.05
DEFAULT_DISCR_LEVEL = 10
####################################

def save_benchmark(benchmarks: Dict[str, float], file_path: str):
    with open(file_path, 'w') as file:
        json.dump(benchmarks, file)

def compute_metrics(all_visisted_positions: np.ndarray, successes, tracks: List[DiscretizedTrajectory], n_discr_level=int(1e4)):

    means = []
    max_devs = []
    n_fails = 0
    for idx, success in enumerate(tqdm(successes)):
        
        if success:
            visited_positions = all_visisted_positions[idx - n_fails]
            track = [wp for wp in tracks[idx]]
            high_discr_ref_traj = TrajectoryFactory.get_pol_discretized_trajectory(
                t_waypoints=track,
                n_points_discretization_level=n_discr_level
            )
            ref_wps = np.array([wp.coordinate for wp in high_discr_ref_traj])
            
            # metrics
            time = len(visited_positions)

            # Compute norms
            # Reshape A and B for broadcasting, compute difference, norm, then mean across axis=1 (corresponding to M)
            norms: np.ndarray = np.linalg.norm(visited_positions[:, np.newaxis, :] - ref_wps[np.newaxis, :, :], axis=2)
            min_distances = norms.min(axis=1)
            mean_dist = np.mean(min_distances)
            
            max_dist = np.max(min_distances)

            # max_dev_norms = norms.max(axis=1)

            means.append(mean_dist)
            max_devs.append(max_dist)
        else:
            n_fails += 1
    return means, max_devs
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
    initial_xyzs = np.array([[0.,     0.,     1.]])
    pts = np.vstack([initial_xyzs, z_segment, y_segment, x_segment])
    t_wps = TrajectoryFactory.waypoints_from_numpy(
        pts
    )
    t_traj = TrajectoryFactory.get_discr_from_wps(t_wps)
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
        discr_level: float=DEFAULT_DISCR_LEVEL
    ):

    eval_set_folder = "./eval-v0_n-ctrl-points-3_n-tracks-10000_2024-02-11_21:49:10_d526388c-3aa6-469e-b354-2f390c2a30c0"
    tracks = load_eval_tracks(eval_set_folder, discr_level=discr_level)

    

    # random number in range 10-99
    output_folder = f"{output_folder}/wp_b={waypoint_buffer_size}_k_p={k_p}_k_wp={k_wp}_k_s={k_s}_max_reward_distance={max_reward_distance}_waypoint_dist_tol={waypoint_dist_tol}"
    print(f"Output folder: {output_folder}")

    all_visited_positions = []
    successes = []
    times = []
    for track in tracks:
        t_traj, init_wp = track, np.array([track[0].coordinate])
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
            single_traj=True,
            eval_mode=True
        )
        visited_positions, success, time = run_test(config=config,
                    env_factory=env_factory, eval_mode=True)
        successes.append(success)
        if success:
            all_visited_positions.append(visited_positions)
            times.append(time)
    
    mean_dev, max_dev = compute_metrics(all_visited_positions, successes, tracks)
    print("SUCCESS RATE: ", np.mean(np.array(successes)))
    print("AVERAGE MEAN DEVIATION: ", np.mean(mean_dev))
    print("AVERAGE MAX DEVIATION: ", np.mean(max_dev))
    print("AVERAGE TIME UNTIL LANDING: ", np.mean(times))

    save_benchmark({
        "success_rate": np.mean(successes),
        "avg mean dev": np.mean(mean_dev),
        "avg max dev": np.mean(max_dev),
        "avt time": np.mean(times)
    }, 
    f'rl_{discr_level}.json')

            


    

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
    parser.add_argument('--discr_level',              default=DEFAULT_DISCR_LEVEL, type=int,           help='Whether example is being run by a notebook (default: "False")', metavar='')

    ARGS = parser.parse_args()

    run(**vars(ARGS))
