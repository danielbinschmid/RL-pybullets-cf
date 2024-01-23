from .waypoint import Waypoint
from typing import List
from .trajectory import Trajectory

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
    

class DiscreteTrajectoryFromContinuous:
    _n_discretization_level: int
    _cont_trajectory: Trajectory

    def __init__(self, cont_traj: Trajectory, n_discretization_level: int=100) -> None:
        self._n_discretization_level = n_discretization_level        
        self._cont_trajectory = cont_traj

    def __len__(self) -> int: 
        return self._n_discretization_level

    def __getitem__(self, idx: int) -> Waypoint:
        t = float(idx) / float(self._n_discretization_level)
        wp = self._cont_trajectory.get_waypoint(t)
        return wp
    
class DiscretizedTrajectoryFromWaypoints:
    wps: List[Waypoint]

    def __init__(self, wps: List[Waypoint]) -> None:
        self.wps = wps

    def __len__(self) -> int:
        return len(self.wps)

    def __getitem__(self, idx: int) -> Waypoint:
        return self.wps[idx]