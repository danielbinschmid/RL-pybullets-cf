from .discretized_trajectory import DiscretizedTrajectory
from .waypoint import Waypoint
from .traj_gen_cpp_wrapper import calculate_trajectory

from typing import List
import numpy as np

def calc_target_durations(waypoints: List[Waypoint]) -> np.ndarray:
    wps = iter(waypoints)
    cur_time = next(wps).timestamp
    assert(cur_time == 0)
    durations = []
    for wp in wps:
        durations.append(wp.timestamp - cur_time)
        cur_time = wp.timestamp

    return np.array(durations)

def convert_to_np_waypoints(waypoints: List[Waypoint]) -> np.ndarray:
    t_waypoints_np = np.zeros((3, len(waypoints)), dtype=np.float64)

    for idx, wp in enumerate(waypoints):
        t_waypoints_np[:,idx] = wp.coordinate
    
    return t_waypoints_np

class PolynomialDiscretizedTrajectory(DiscretizedTrajectory):
    waypoints: np.ndarray
    timestamps: np.ndarray

    def __init__(self, t_waypoints: List[Waypoint], n_points_discretization_level: int) -> None:
        
        # target waypoints with durations
        t_waypoints_np = convert_to_np_waypoints(t_waypoints)
        t_durations_np = calc_target_durations(t_waypoints)

        # call cpp backbone
        self.waypoints, self.timestamps = calculate_trajectory(
            t_waypoints_np,
            t_durations_np,
            n_points_discretization_level
        )
                
        # assert n_points
        assert(self.waypoints.shape[1] == n_points_discretization_level) 
        assert(self.waypoints.shape[1] == len(self.timestamps))

    def __len__(self) -> int:
        return len(self.timestamps)        

    def __getitem__(self, idx: int) -> Waypoint:
        wp = Waypoint(
            self.waypoints[:, idx],
            self.timestamps[idx]
        )
        return wp

    def _reverse_timestamps(self):
        min_ = np.min(self.timestamps)
        max_ = np.max(self.timestamps)
        t_reversed = np.array([min_ + (max_ - t) for t in self.timestamps])
        self.timestamps = t_reversed

    def reverse(self):
        """
        Reverses trajectory in-place
        """
        self.waypoints = np.flip(self.waypoints)
        self._reverse_timestamps()
