def test_imports():
    from trajectories import TrajectoryFactory

def test_square_trajectory():
    from trajectories import TrajectoryFactory 
    from trajectories.trajectories import SquareLinearTrajectory
    import numpy as np
    trajectory: SquareLinearTrajectory = TrajectoryFactory.get_linear_square_trajectory()
    p0 = trajectory.get_waypoint(0) 
    p1 = trajectory.get_waypoint(0.25)
    p2 = trajectory.get_waypoint(0.5)
    p3 = trajectory.get_waypoint(0.75)

    # assert corner points
    assert((p0.coordinate == trajectory.corner_points[0]).all())
    assert((p1.coordinate == trajectory.corner_points[1]).all())
    assert((p2.coordinate == trajectory.corner_points[2]).all())
    assert((p3.coordinate == trajectory.corner_points[3]).all())

    # assert interpolation
    assert(trajectory.get_waypoint(0.1).coordinate == np.asarray([0.4, 0., 1.], dtype=np.float32)).all()

