import sys

sys.path.append("../..")
import numpy as np
import runnables.visualization.utils.plotting as plotting
import matplotlib.pyplot as plt
from trajectories import DiscretizedTrajectory, Waypoint, TrajectoryFactory
from typing import List, Tuple
import numpy as np
from gym_pybullet_drones.utils.pos_logger import load_positions
import matplotlib.cm as cm
import matplotlib.lines as mlines
from runnables.utils.gen_eval_tracks import load_eval_tracks


def wps_to_ndarray(wps: List[Waypoint]):
    arr = np.zeros((3, len(wps)))

    for idx, wp in enumerate(wps):
        arr[:, idx] = wp.coordinate

    return arr


def discr_traj_to_ndarray(traj: DiscretizedTrajectory) -> np.ndarray:
    wps = []
    for idx in range(len(traj)):
        wps.append(traj[idx])
    return wps_to_ndarray(wps)


color_boutique = ["#1f77b4", "black", "red", "green", "#ff7f0e"]

cmaps = [cm.cividis, cm.pink, cm.bone, cm.summer]

marker = ["x", "."]

names = ["Position RL", "Position PID", "Trajectory RL"]


def vis_discr_traj(
    trajs_: List[Tuple[DiscretizedTrajectory, np.ndarray]],
    wps: DiscretizedTrajectory = None,
    plotname: str = "minimum_snap_poly_traj.pdf",
    label_: str = "PID",
):
    plotting.prepare_for_latex()

    # print(trajs[0][1])
    trajs = []
    vels = []
    for x in trajs_:
        traj, vel = x
        trajs.append(traj)
        vels.append(vel)

    vels = [np.linalg.norm(vel, axis=1) for vel in vels]

    max_length = max([len(traj) for traj in trajs])
    time_sequence = np.linspace(0, 0.5, max_length)[::-1]

    # Create a 3D plot
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    trajectory_color = "#1f77b4"  # A pleasant shade of blue
    waypoint_color = "black"  # A contrasting shade of orange
    # Plot the trajectory
    # ax.plot(x, y, z, label='3D Trajectory', color='b', linewidth=2)
    for idx, traj in enumerate(trajs):
        traj_np = discr_traj_to_ndarray(traj)

        x = traj_np[0]
        y = traj_np[1]
        z = traj_np[2]
        time_seq_traj = time_sequence[: len(traj)]
        color_map = cmaps[idx](vels[idx])
        ax.scatter(x, y, z, c=color_map, marker=".", s=15, label=f"{idx}")

    if wps is not None:
        wps_np = discr_traj_to_ndarray(wps)
        reference_x = wps_np[0]
        reference_y = wps_np[1]
        reference_z = wps_np[2]
        print(reference_x)
        print(reference_y)
        print(reference_z)
        # Plot the additional waypoints
        ax.scatter(
            reference_x,
            reference_y,
            reference_z,
            color=waypoint_color,
            marker="x",
            s=20,
            label="Target waypoints",
        )

    xmin = min([pos.coordinate[0] for positions in trajs for pos in positions])
    xmax = max([pos.coordinate[0] for positions in trajs for pos in positions])
    ymin = min([pos.coordinate[1] for positions in trajs for pos in positions])
    ymax = max([pos.coordinate[1] for positions in trajs for pos in positions])
    zmin = min([pos.coordinate[2] for positions in trajs for pos in positions])
    zmax = max([pos.coordinate[2] for positions in trajs for pos in positions])

    print(xmin)
    print(xmax)
    # Set the axes limits to 0-100
    ax.set_xlim([-0.3, 1.7])
    ax.set_ylim([-0.1, 0.3])
    ax.set_zlim([0, 1.6])

    # Customization
    ax.set_title("")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=cmaps[0]([0.95]),
            marker="o",
            linestyle="None",
            markersize=5,
            label=label_,
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="x",
            linestyle="None",
            markersize=5,
            label="Target waypoints",
        ),
    ]
    ax.legend(handles=legend_handles)

    # Create custom legend
    # plt.legend(handles=legend_handles, loc='best')

    # Display the plot
    plt.savefig(plotname)


def load_from_path(path: str):
    visited_positions, veloctities = load_positions(log_folder=path)
    wps = TrajectoryFactory.waypoints_from_numpy(visited_positions)
    traj = TrajectoryFactory.get_discr_from_wps(t_waypoints=wps)
    return traj, veloctities


def vis():
    tracks = load_eval_tracks(
        "./test_tracks/eval-v0_n-ctrl-points-3_n-tracks-200_2024-02-12_12:21:14_3115f5a9-bd48-4b63-9194-5eda43e1329d",
        discr_level=9,
    )
    track = tracks[0]

    traj_pid, velocities = load_from_path("./visualization/logs/landing_pid")
    trajs = [(traj_pid, velocities)]
    vis_discr_traj(trajs, track, plotname="landing_pid.pdf", label_="PID")

    traj_pid, velocities = load_from_path("./visualization/logs/landing_rl")
    trajs = [(traj_pid, velocities)]
    vis_discr_traj(trajs, track, plotname="landing_rl.pdf", label_="RL")


if __name__ == "__main__":
    vis()
