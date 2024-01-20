from trajectory_cpp import calc_trajectory
import numpy as np
from typing import Tuple

def calculate_trajectory(
    t_waypoints: np.ndarray,
    t_durations: np.ndarray,
    n_discretization_level: int
) -> Tuple[np.ndarray, np.ndarray]:
    r_waypoints = np.zeros((3, n_discretization_level), dtype=np.float64)
    r_timestamps = np.zeros(n_discretization_level, dtype=np.float64)
    calc_trajectory(
        t_waypoints,
        t_durations,
        r_waypoints,
        r_timestamps
    )
    return r_waypoints, r_timestamps
    