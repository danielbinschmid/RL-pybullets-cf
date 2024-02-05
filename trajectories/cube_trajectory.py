from .waypoint import Waypoint
from .trajectory import Trajectory
import numpy as np


class CubeTrajectory(Trajectory):

    def __init__(self) -> None:
        self.waypoints = [
            Waypoint(np.array([0.0,   0.0,   0.6]), 0),
            Waypoint(np.array([1.0,   0.0,   0.6]), 1),
            Waypoint(np.array([1.0,   1.0,   0.6]), 2),
            Waypoint(np.array([0.0,   1.0,   0.6]), 3),
            Waypoint(np.array([0.0,   0.0,   0.6]), 4),
            Waypoint(np.array([0.0,   0.0,   0.2]), 5),
            Waypoint(np.array([1.0,   0.0,   0.2]), 6),
            Waypoint(np.array([1.0,   1.0,   0.2]), 7),
            Waypoint(np.array([0.0,   1.0,   0.2]), 8)
        ]
        self.counter = 0

    def get_waypoint(self, time: float) -> Waypoint:
        """
        Trajectory drawing a cube of side length 1 at height 0.5
        """
        waypoint = self.waypoints[time % len(self.waypoints)]
        return waypoint

    def get_next_waypoint(self) -> Waypoint:
        """
        Trajectory drawing a cube of side length 1 at height 0.5, time is meaningless
        """
        waypoint = self.waypoints[self.counter % len(self.waypoints)]
        self.counter += 1
        return waypoint

    def reset(self):
        self.counter = 0

    def is_done(self):
        return self.counter >= len(self.waypoints)
