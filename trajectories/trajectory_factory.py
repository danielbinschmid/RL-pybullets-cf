from .trajectory import Trajectory
from .square_linear_trajectory import SquareLinearTrajectory
from .polynomial_discretized_trajectory import PolynomialDiscretizedTrajectory
from .waypoint import Waypoint
from typing import List
import numpy as np 
from .discretized_trajectory import DiscretizedTrajectory, DiscreteTrajectoryFromContinuous, DiscretizedTrajectoryFromWaypoints
from typing import Optional

class WAYPOINT_BOOTH:
    T_WAYPOINTS_POLY = [
         Waypoint(
              coordinate=np.asarray([0, 0, 0.25]),
              timestamp=1
         ),
         Waypoint(
              coordinate=np.asarray([0, 0, 0.25]),
              timestamp=2
         ),
         Waypoint(
              coordinate=np.asarray([0, 0, 0.5]),
              timestamp=3
         ),
         Waypoint(
              coordinate=np.asarray([0, 0, 0.25]),
              timestamp=4
         ),
         Waypoint(
              coordinate=np.asarray([0, 0, 0]),
              timestamp=5
         )
    ]
    T_WAYPOINTS_POLY_ALL = [
         Waypoint(
              coordinate=np.asarray([0, 0, 0]),
              timestamp=0
         ),
         Waypoint(
              coordinate=np.asarray([0, 0.5, 0.15]),
              timestamp=1
         ),
         Waypoint(
              coordinate=np.asarray([0.25, 0.25, 0.075]),
              timestamp=2
         ),
         Waypoint(
              coordinate=np.asarray([0, 0, 0]),
              timestamp=3
         )
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
    def get_linear_square_traj_discretized(cls, n_discretization_level: int=100, square_scale: float=1, time_scale: float=1) -> DiscretizedTrajectory:
          sq_traj = SquareLinearTrajectory(
              square_scale, time_scale
          )
          discr_traj = DiscreteTrajectoryFromContinuous(sq_traj, n_discretization_level) 
          return discr_traj

    @classmethod
    def get_pol_discretized_trajectory(cls, t_waypoints: Optional[List[Waypoint]]=None, n_points_discretization_level: Optional[int] = None) -> PolynomialDiscretizedTrajectory:
        
        if t_waypoints is None:
            t_waypoints = WAYPOINT_BOOTH.T_WAYPOINTS_POLY_ALL
        if n_points_discretization_level is None: 
            n_points_discretization_level = 100
        
        return PolynomialDiscretizedTrajectory(
            t_waypoints,
            n_points_discretization_level
        )
    
    @classmethod
    def get_simple_smooth_trajectory(cls, starting_waypoint: Waypoint, t_waypoints: Optional[List[Waypoint]]=None, n_points_discretization_level: Optional[int] = None) -> DiscretizedTrajectory:
        if t_waypoints is None:
            t_waypoints = [starting_waypoint] + WAYPOINT_BOOTH.T_WAYPOINTS_POLY
        if n_points_discretization_level is None: 
            n_points_discretization_level = 100
        
        return PolynomialDiscretizedTrajectory(
            t_waypoints,
            n_points_discretization_level
        ) 
    
    @classmethod
    def get_discr_from_wps(cls, t_waypoints: List[Waypoint]) -> DiscretizedTrajectory:
        traj = DiscretizedTrajectoryFromWaypoints(t_waypoints)        
        return traj
    
    @classmethod
    def waypoints_from_numpy(cls, t_waypoints: np.ndarray) -> List[Waypoint]:
        res = [
            Waypoint(
                t_waypoints[idx],
                timestamp=idx
            )
            for idx 
            in range(len(t_waypoints))
        ]
        return res
