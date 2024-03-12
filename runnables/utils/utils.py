import argparse
import numpy as np
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import TrajectoryFactory, DiscretizedTrajectory
from agents.utils.configuration import Configuration
from aviaries.factories.uzh_trajectory_follower_factory import TrajectoryFollowerAviaryFactory

from agents.test_policy import run_test
from agents.train_policy import run_train
from runnables.evaluation.gen_eval_tracks import load_eval_tracks 
from torch import nn
from typing import List, Dict
from tqdm import tqdm
import json 

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

def compute_metrics_single(visited_positions: np.ndarray, track: DiscretizedTrajectory, n_discr_level=int(1e4)):


    track = [wp for wp in track]
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

    return mean_dist, max_dist