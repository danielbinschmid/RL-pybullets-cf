from .trajectory import Trajectory
from .square_linear_trajectory import SquareLinearTrajectory
from .polynomial_discretized_trajectory import PolynomialDiscretizedTrajectory
from .waypoint import Waypoint
from typing import List

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
    def get_pol_discretized_trajectory(cls, t_waypoints: List[Waypoint], n_points_discretization_level: int) -> PolynomialDiscretizedTrajectory:
        return PolynomialDiscretizedTrajectory(
            t_waypoints,
            n_points_discretization_level
        )

