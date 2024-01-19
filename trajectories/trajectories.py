from .waypoint import Waypoint
import numpy as np 
import math 

class Trajectory:

    def __init__(self) -> None:
        pass

    def get_waypoint(self, time: float) -> Waypoint:
        """
        Needs to be implemented by subclass. time \in [0,1]
        """
        raise NotImplementedError()

class SquareLinearTrajectory(Trajectory):
    """
    Linear interpolation for four corner points of a 2D square.
    """

    square_scale: float 
    time_scale: float
    corner_points: np.ndarray

    def __init__(self, square_scale: float=1, time_scale: float=1) -> None:
        super().__init__()
        self.square_scale = square_scale
        self.time_scale = time_scale
        self.corner_points = np.array([
            [0                  ,0                  ,self.square_scale],
            [self.square_scale  ,0                  ,self.square_scale],
            [self.square_scale  ,self.square_scale  ,self.square_scale],
            [0                  ,self.square_scale  ,self.square_scale],
        ], np.float32)
        
    def get_waypoint(self, time: float):
        assert time >= 0 and time <= 1
        time = time * 4
        
        if int(time) == time:
            # exactly at corner point
            target_pos = self.corner_points[int(time)]
        else:
            # in-between two points, linear interpolation
            cur_corner = math.floor(time)
            upcomimg_corner = math.ceil(time)
            diff = upcomimg_corner - cur_corner
            target_pos = cur_corner + (time - int(time)) * (diff)

        target_wp = Waypoint(
            coordinate=target_pos,
            timestamp=time * self.time_scale
        )

        return target_wp


    