from .waypoint import Waypoint
from typing import List
from .trajectory import Trajectory
import numpy as np

class DiscretizedTrajectory:

    def __init__(self) -> None:
        pass

    def __len__(self) -> int: 
        """
        Number of waypoints of discretized trajectory.

        Must be implemented by child class.
        """
        raise NotImplementedError("Must be implemented by child class.")
        
    def __getitem__(self, idx: int) -> Waypoint:
        """
        Yields waypoint with index 'idx'.

        Must be implemented by child class.
        """
        raise NotImplementedError("Must be implemented by child class.")
    
    def export_to_np(self, fname: str):
        """
        Limitation: Does not export timestamp information.
        """
        wps = np.concatenate([self[i].coordinate for i in range(self.__len__())])
        np.save(fname, wps)

    def __str__(self) -> str:
        log_str = ""
        for i in range(self.__len__()):
            wp = self.__getitem__(i)
            wp_log_str = f'Waypoint {i}, coordinate: {wp.coordinate}; timestamp: {wp.timestamp} \n'
            log_str += wp_log_str
        return log_str

class DiscreteTrajectoryFromContinuous(DiscretizedTrajectory):
    _n_discretization_level: int
    _cont_trajectory: Trajectory

    def __init__(self, cont_traj: Trajectory, n_discretization_level: int=100) -> None:
        super().__init__()
        self._n_discretization_level = n_discretization_level        
        self._cont_trajectory = cont_traj

    def __len__(self) -> int: 
        return self._n_discretization_level

    def __getitem__(self, idx: int) -> Waypoint:
        t = float(idx) / float(self._n_discretization_level)
        wp = self._cont_trajectory.get_waypoint(t)
        return wp
    
class DiscretizedTrajectoryFromWaypoints(DiscretizedTrajectory):
    wps: List[Waypoint]

    def __init__(self, wps: List[Waypoint]) -> None:
        
        self.wps = wps

    def __len__(self) -> int:
        return len(self.wps)

    def __getitem__(self, idx: int) -> Waypoint:
        return self.wps[idx]
    
    def reverse(self):
        """
        Reverses trajectory in-place
        """
        self.wps = np.flip(self.wps)
    
class DiscretizedTrajFromNumpy(DiscretizedTrajectoryFromWaypoints):
    def __init__(self, wps_np: np.ndarray) -> None:
        wps_np = wps_np.reshape((-1, 3))
        wps = [
            Waypoint(
                coordinate=wp,
                timestamp=i 
            )
            for i, wp in enumerate(wps_np)
        ]
        super().__init__(wps=wps)