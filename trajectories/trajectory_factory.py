from time import thread_time
import numpy as np

class Waypoint:
    """
    3D coordinate with a timestamp.
    """
    
    coordinate: np.ndarray
    """R^3 coordinate"""
    timestamp: float
    """Time as float"""

    def __init__(self, coordinate: np.ndarray, timestamp: float) -> None:
        self.coordinate = coordinate
        self.timestamp = timestamp


class TrajectoryFactory:
    """
    Generates waypoints
    """
    
    def __init__(self) -> None:
        pass

    def get_waypoint(self) -> Waypoint:
        waypoint = Waypoint(
            coordinate=np.asarray([0,0,0]),
            timestamp=0
        )
        return waypoint

