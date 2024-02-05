from .waypoint import Waypoint
from .trajectory import Trajectory
import numpy as np


class RandomTrajectory(Trajectory):

    def __init__(self) -> None:
        pass

    def get_next_waypoint(self) -> Waypoint:
        """
        Returns a random point in the cube of side length 1, timestamp is meaningless
        """
        point = np.random.uniform(low=0, high=1, size=(3,))
        return Waypoint(point, None)

    def reset(self):
        pass

    def is_done(self):
        return False
