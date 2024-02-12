import sys 
sys.path.append("../..")
from trajectories import TrajectoryFactory, DiscretizedTrajectory
import numpy as np 
from uuid import uuid4
import os 
import shutil
from datetime import datetime
from typing import List 
from tqdm import tqdm 
def gen_eval_tracks(target_folder: str, n_tracks: int, n_ctrl_points: int=3):
    
    # target folder
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    target_folder = target_folder + f'_n-ctrl-points-{n_ctrl_points}_n-tracks-{n_tracks}_{datetime_string}_{str(uuid4())}' 
    if os.path.isfile(target_folder):
        os.remove(target_folder)
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)

    for i in tqdm(range(n_tracks)):
        _, ctrl_wp = TrajectoryFactory.gen_random_trajectory(
            start=np.array([0, 0, 0]),
            n_discr_level=10,
            n_ctrl_points=n_ctrl_points,
            std_dev_deg=50,
            distance_between_ctrl_points=1.3,
            init_dir=None,
            return_ctrl_points=True
        )
        traj = TrajectoryFactory.get_discr_from_wps(ctrl_wp)
        traj.reverse()
        traj_folder = os.path.join(
            target_folder, f'{i}.npy'
        )
        traj.export_to_np(traj_folder)

def load_eval_tracks(folder: str, discr_level: int=10) -> List[DiscretizedTrajectory]:
    files = os.listdir(folder)
    trajs = []
    print("Loading test tracks..")
    for fname in tqdm(files):
        fpath = os.path.join(folder, fname)
        traj = TrajectoryFactory.get_discr_from_np(fpath)
        wps = [x for x in traj]
        traj_poly = TrajectoryFactory.get_pol_discretized_trajectory(
            t_waypoints=wps, 
            n_points_discretization_level=discr_level
        )
        trajs.append(traj_poly)
    print("Loading test tracks done.")
    return trajs

if __name__ == "__main__":
    gen = True
    load = False
    load_folder = "./eval-v0_n-ctrl-points-5_n-tracks-10000_2024-02-11_20:15:45_151aebc7-4bca-422f-a63c-44c55d53bda5"

    if gen:
        t_folder = "./eval-v0"
        n_tracks = 1000
        gen_eval_tracks(t_folder, n_tracks=n_tracks)
    if load: 
        tracks = load_eval_tracks(load_folder)
        for track in tracks:
            print(track)


    