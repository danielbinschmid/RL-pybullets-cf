from .trajectory import Trajectory
import numpy as np
from .waypoint import Waypoint
import math


class SquareLinearTrajectory(Trajectory):
    """
    Linear interpolation for four corner points of a 2D square.
    """

    square_scale: float
    time_scale: float
    corner_points: np.ndarray

    def __init__(self, square_scale: float = 1, time_scale: float = 1) -> None:
        super().__init__()
        self.square_scale = square_scale
        self.time_scale = time_scale
        self.corner_points = np.array(
            [
                [0, 0, self.square_scale],
                [self.square_scale, 0, self.square_scale],
                [self.square_scale, self.square_scale, self.square_scale],
                [0, self.square_scale, self.square_scale],
            ],
            np.float32,
        )

    def get_waypoint(self, time: float):
        assert time >= 0 and time <= 1
        time = time * 4

        if int(time) == time:
            # exactly at corner point
            target_pos = self.corner_points[int(time)]
        else:
            # in-between two points, linear interpolation
            cur_corner = math.floor(time) % 4
            upcomimg_corner = math.ceil(time) % 4
            diff = self.corner_points[upcomimg_corner] - self.corner_points[cur_corner]
            target_pos = self.corner_points[cur_corner] + (time - int(time)) * (diff)

        target_wp = Waypoint(
            coordinate=target_pos, timestamp=(time * self.time_scale) / 4
        )

        return target_wp
