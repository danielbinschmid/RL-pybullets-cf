import numpy as np
from typing import Optional

class TrajectoryQuery:
    t_waypoints: np.ndarray
    t_durations: np.ndarray
    r_waypoints: np.ndarray
    r_timestamps: np.ndarray

    def __init__(self, 
                    t_waypoints: np.ndarray,
                    t_durations: np.ndarray,
                    t_discretization_level: int
    ) -> None:
        self.t_waypoints = t_waypoints
        self.t_durations = t_durations
        self.r_waypoints = np.zeros((3, t_discretization_level), dtype=np.float64)
        self.r_timestamps = np.zeros(t_discretization_level, dtype=np.float64)
