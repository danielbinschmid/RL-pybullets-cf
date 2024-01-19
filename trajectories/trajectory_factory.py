from .trajectories import Trajectory, SquareLinearTrajectory

class TrajectoryFactory:
    """
    Wrapper class for instantiating target trajectories.
    """

    @classmethod
    def get_linear_square_trajectory(cls, square_scale: float, time_scale: float) -> Trajectory:
        return SquareLinearTrajectory(
            square_scale, time_scale
        )


