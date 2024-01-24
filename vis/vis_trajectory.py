import sys
sys.path.append("..")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotting
from vis.utils.generate_trajectory import generate

plotting.prepare_for_latex()
# Create data points for the trajectory
traj_query = generate()

wps = traj_query.r_waypoints
x = wps[0]
y = wps[1]
z = wps[2]

reference_x = traj_query.t_waypoints[0]
reference_y = traj_query.t_waypoints[1]
reference_z = traj_query.t_waypoints[2]



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


# Customization
ax.set_title("3D Trajectory Plot")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.legend()

# Display the plot
plt.savefig("generated_polynomial_trajectory.pdf")
