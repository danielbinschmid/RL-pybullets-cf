
def test_imports():
    from trajectories import TrajectoryFactory

def test_square_trajectory():
    from trajectories import TrajectoryFactory 
    from trajectories.square_linear_trajectory import SquareLinearTrajectory
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

def test_pol_trajectory():
    from trajectories import TrajectoryFactory, Waypoint
    import numpy as np
    t_waypoints = [
        Waypoint(
            np.array([0,0,0], dtype=np.float64),
            0
        ),
        Waypoint(
            np.array([1,1,1], dtype=np.float64),
            1
        ),
        Waypoint(
            np.array([2,2,2], dtype=np.float64),
            2
        ),
        Waypoint(
            np.array([3,3,3], dtype=np.float64),
            3
        )
    ]
    traj = TrajectoryFactory.get_pol_discretized_trajectory(
        t_waypoints,
        10
    )
    threshhold = 0.001
    assert(np.sum(np.square(traj[0].coordinate - t_waypoints[0].coordinate)) < threshhold)
    assert(np.sum(np.square(traj[9].coordinate - t_waypoints[3].coordinate)) < threshhold)
