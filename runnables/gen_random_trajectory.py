import sys 
sys.path.append("..")

from trajectories import TrajectoryFactory
from vis import vis_discr_traj
import numpy as np 

traj, ctrl_wps = TrajectoryFactory.gen_random_trajectory(
    start=np.array([0.1, 0.1, 0.1]),
    n_discr_level=1000,
    n_ctrl_points=20,
    std_dev_deg=90,
    distance_between_ctrl_points=0.1,
    init_dir=np.array([0, 1, 0]),
    return_ctrl_points=True
)
print(traj)
print(ctrl_wps)

vis_discr_traj(traj, ctrl_wps)