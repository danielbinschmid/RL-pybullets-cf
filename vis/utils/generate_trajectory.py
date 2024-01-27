import numpy as np
from trajectory_cpp import calc_trajectory
from typing import Tuple
from .trajectory_query import TrajectoryQuery
from .trajectory_factory import TrajectoryFactory

def generate() -> TrajectoryQuery:
    
    # CONFIG +++++
    LOG = False
    # ++++++++++++

    # SETUP ++++++
    query = TrajectoryFactory.gen_traj_curved()
    # ++++++++++++

    # API ++++++++
    calc_trajectory(
        query.t_waypoints, 
        query.t_durations, 
        query.r_waypoints, 
        query.r_timestamps)
    # ++++++++++++

    # LOG ++++++++
    if LOG:
        print("Target waypoints: ", query.t_waypoints)
        print("Resulting waypoints: ", query.r_waypoints)
    # ++++++++++++

    # RET ++++++++
    return query
    # ++++++++++++
    

if __name__ == "__main__":
    generate() 