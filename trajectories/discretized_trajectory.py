from .waypoint import Waypoint
from typing import List

class DiscretizedTrajectory:
    waypoints: List[Waypoint]

    def __init__(self) -> None:
        pass

    def __len__(self) -> int: 
        """
        Must be implemented by child class.
        """
        raise NotImplementedError("Must be implemented by child class.")
        
    def __getitem__(self, idx: int) -> Waypoint:
        """
        Must be implemented by child class.
        """
        raise NotImplementedError("Must be implemented by child class.")