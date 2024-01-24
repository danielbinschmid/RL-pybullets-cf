from .trajectory_query import TrajectoryQuery
import numpy as np 

class TrajectoryFactory:

    @classmethod
    def gen_traj_debug(cls) -> TrajectoryQuery:
        t_waypoints = np.array([ 
            [0,  9.17827,  14.2934,   9.17827],
            [0, -7.98463, -15.3833,  -7.98463],
            [0,  6.74148, -3.04313,   6.74148]
        ], dtype=np.float64)
        t_durations = np.array([2, 2, 2], dtype=np.float64) 
        return TrajectoryQuery(
            t_waypoints=t_waypoints,
            t_durations=t_durations,
            t_discretization_level=100
        )
    
    @classmethod
    def gen_traj_curved(cls) -> TrajectoryQuery:
        t_waypoints = np.array([ 
            [0,     0,      1,      1,      0, 1],
            [0,     1,      1,      0,      0, 1],
            [0,     0.25,   0.5,    0.75,   1, 1]
        ], dtype=np.float64)
        t_durations = np.array([2, 2, 2, 2, 2], dtype=np.float64)
        return TrajectoryQuery(
            t_waypoints=t_waypoints,
            t_durations=t_durations,
            t_discretization_level=100
        )