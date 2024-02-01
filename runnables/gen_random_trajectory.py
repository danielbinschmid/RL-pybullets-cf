import sys 
sys.path.append("..")

from trajectories import TrajectoryFactory
from vis import vis_discr_traj
import numpy as np 

traj, ctrl_wps = TrajectoryFactory.gen_random_trajectory(
    start=np.array([50, 50, 50]),
    n_discr_level=1000,
    n_ctrl_points=10,
    std_dev_dev=90,
    distance_between_ctrl_points=20,
    init_dir=np.array([0, 1, 0]),
    return_ctrl_points=True
)

vis_discr_traj(traj, ctrl_wps)