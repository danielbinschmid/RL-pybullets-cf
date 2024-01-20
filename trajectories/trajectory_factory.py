from .trajectory import Trajectory
from .square_linear_trajectory import SquareLinearTrajectory
from .polynomial_discretized_trajectory import PolynomialDiscretizedTrajectory
from .waypoint import Waypoint
from typing import List
import numpy as np 
from typing import Optional

class WAYPOINT_BOOTH:
    T_WAYPOINTS_POLY = [
         Waypoint(
              coordinate=np.asarray([0, 0, 0]),
              timestamp=0
         ),
         Waypoint(
              coordinate=np.asarray([0, 1, 0.25]),
              timestamp=2
         ),
         Waypoint(
              coordinate=np.asarray([1, 1, 0.5]),
              timestamp=4
         ),
         Waypoint(
              coordinate=np.asarray([1, 0, 0.75]),
              timestamp=8
         ),
         Waypoint(
              coordinate=np.asarray([0, 0, 1]),
              timestamp=10
         ), 
         Waypoint(
              coordinate=np.asarray([1,1,1]),
              timestamp=12
         ),
         Waypoint(
              coordinate=np.asarray([1, 0, 0.75]),
              timestamp=14
         ),
         Waypoint(
              coordinate=np.asarray([1, 1, 0.5]),
              timestamp=16
         ),
         Waypoint(
              coordinate=np.asarray([0, 0, 0]),
              timestamp=18
         ),
    ]


class TrajectoryFactory:
    """
    Wrapper class for instantiating target trajectories.
    """

    @classmethod
    def get_linear_square_trajectory(cls, square_scale: float=1, time_scale: float=1) -> SquareLinearTrajectory:
        return SquareLinearTrajectory(
            square_scale, time_scale
        )

    @classmethod
    def get_pol_discretized_trajectory(cls, t_waypoints: Optional[List[Waypoint]]=None, n_points_discretization_level: Optional[int] = None) -> PolynomialDiscretizedTrajectory:
        
        if t_waypoints is None:
            t_waypoints = WAYPOINT_BOOTH.T_WAYPOINTS_POLY
        if n_points_discretization_level is None: 
            n_points_discretization_level = 100
        
        return PolynomialDiscretizedTrajectory(
            t_waypoints,
            n_points_discretization_level
        )

