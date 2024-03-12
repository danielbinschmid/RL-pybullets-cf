import runnables.visualization.utils.plotting as plotting
import matplotlib.pyplot as plt
from trajectories import DiscretizedTrajectory, Waypoint
from typing import List
import numpy as np 

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

def vis_discr_traj(traj: DiscretizedTrajectory, wps: List[Waypoint], plotname: str="minimum_snap_poly_traj.pdf"):
    plotting.prepare_for_latex()

    traj_np = discr_traj_to_ndarray(traj)
    wps_np = wps_to_ndarray(wps)

    x = traj_np[0]
    y = traj_np[1]
    z = traj_np[2]

    reference_x = wps_np[0]
    reference_y = wps_np[1]
    reference_z = wps_np[2]



    # Create a 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    trajectory_color = "#1f77b4"  # A pleasant shade of blue
    waypoint_color = "#ff7f0e"    # A contrasting shade of orange
    # Plot the trajectory
    # ax.plot(x, y, z, label='3D Trajectory', color='b', linewidth=2)
    ax.scatter(x, y, z, color=trajectory_color, marker='.', s=20, label='Generated polynomial trajectory')

    # Plot the additional waypoints
    ax.scatter(reference_x, reference_y, reference_z, color=waypoint_color, marker='x', s=100, label='Target waypoints')

    # Set the axes limits to 0-100
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])

    # Customization
    ax.set_title("3D Trajectory Plot")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.legend()

    # Display the plot
    plt.savefig(plotname)