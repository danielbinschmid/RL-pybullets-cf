import sys 
sys.path.append("../..")
import numpy as np
import vis.plotting as plotting
import matplotlib.pyplot as plt
from trajectories import DiscretizedTrajectory, Waypoint, TrajectoryFactory
from typing import List
import numpy as np 
from gym_pybullet_drones.utils.pos_logger import load_positions
import matplotlib.cm as cm
import matplotlib.lines as mlines


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

color_boutique = [
    "#1f77b4",
    "black",
    "red",
    "green",
    "#ff7f0e"
]

cmaps = [
    cm.pink,
    cm.bone,
    cm.summer
]

marker = [
    "x",
    "."
]

names = [
    "Position RL",
    "Position PID",
    "Trajectory RL"
]

def vis_discr_traj(trajs: List[DiscretizedTrajectory], wps: DiscretizedTrajectory=None, plotname: str="minimum_snap_poly_traj.pdf"):
    plotting.prepare_for_latex()

    max_length = max([len(traj) for traj in trajs])
    time_sequence = np.linspace(0, 0.5, max_length)[::-1]

    # Create a 3D plot
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    trajectory_color = "#1f77b4"  # A pleasant shade of blue
    waypoint_color = "black"    # A contrasting shade of orange
    # Plot the trajectory
    # ax.plot(x, y, z, label='3D Trajectory', color='b', linewidth=2)
    for idx, traj in enumerate(trajs):
        traj_np = discr_traj_to_ndarray(traj)

        x = traj_np[0]
        y = traj_np[1]
        z = traj_np[2]
        time_seq_traj = time_sequence[:len(traj)]
        color_map = cmaps[idx](time_seq_traj)
        ax.scatter(x, y, z, c=color_map, marker=".", s=10, label=f'{idx}')
        
    if wps is not None: 
        wps_np = discr_traj_to_ndarray(wps)
        reference_x = wps_np[0]
        reference_y = wps_np[1]
        reference_z = wps_np[2]
        print(reference_x)
        print(reference_y)
        print(reference_z)
        # Plot the additional waypoints
        ax.scatter(reference_x, reference_y, reference_z, color=waypoint_color, marker='x', s=100, label='Target waypoints')


    
    # Set the axes limits to 0-100
    ax.set_xlim([-0.3, 1.3])
    ax.set_ylim([-0.3, 1.3])
    ax.set_zlim([-0.3, 1.3])

    # Customization
    ax.set_title("")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    legend_handles = [
        mlines.Line2D([], [], color=cmaps[0]([0.5]), marker='o', linestyle='None', markersize=10, label=names[0]),
        mlines.Line2D([], [], color=cmaps[1]([0.5]), marker='o', linestyle='None', markersize=10, label=names[1])
    ]
    ax.legend(handles=legend_handles)

    

    # Create custom legend
    # plt.legend(handles=legend_handles, loc='best')


    # Display the plot
    plt.savefig(plotname)

def load_from_path(path:str):
    visited_positions = load_positions(
        log_folder=path
    )
    wps = TrajectoryFactory.waypoints_from_numpy(visited_positions)
    traj = TrajectoryFactory.get_discr_from_wps(
        t_waypoints=wps
    )
    return traj

def vis():
    traj_rl = load_from_path("/shared/d/master/ws23/UAV-lab/git_repos/RL-pybullets-cf/runnables/test_suite/logs/pos_logs_rl")
    traj_pid = load_from_path("/shared/d/master/ws23/UAV-lab/git_repos/RL-pybullets-cf/runnables/test_suite/logs/pos_logs_pid")
    # traj_rl_traj = load_from_path("/shared/d/master/ws23/UAV-lab/git_repos/RL-pybullets-cf/runnables/test_suite/logs/pos_logs_rl_traj")

    trajs = [traj_rl, traj_pid ]

    t_traj = TrajectoryFactory.get_linear_square_traj_discretized(
        n_discretization_level=4,
    )
    vis_discr_traj(trajs, t_traj, plotname="wp_following.pdf")




if __name__ == "__main__":
    vis()