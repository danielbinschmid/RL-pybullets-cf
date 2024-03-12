import sys

sys.path.append("..")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import runnables.visualization.utils.plotting as plotting

plotting.prepare_for_latex()


def generate_random_vector_in_cone(original_vector, cone_angle_deg, std_deviation):
    """
    Generates a random direction unit vector within a cone around an original vector.
    The distribution of directions is Gaussian around the cone's axis.

    Parameters:
    - original_vector: The unit vector around which the cone is centered.
    - cone_angle_deg: The angle of the cone in degrees.
    - std_deviation: Standard deviation of the Gaussian distribution for the angle.

    Returns:
    - A random unit vector within the cone.
    """

    # Convert cone angle from degrees to radians
    cone_angle_rad = np.radians(cone_angle_deg)

    # Generate a random angle θ from the Gaussian distribution
    theta = np.random.normal(0, std_deviation, 1)[0]
    theta = min(max(theta, 0), cone_angle_rad)  # Clamp θ within [0, cone_angle]

    # Generate a random azimuthal angle φ uniformly between 0 and 2π
    phi = np.random.uniform(0, 2 * np.pi)

    # Convert spherical coordinates to Cartesian coordinates (in aligned system)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Construct a unit vector in the aligned coordinate system
    random_vector_aligned = np.array([x, y, z])

    # Align original_vector with the z-axis
    # This example assumes a simple alignment for demonstration. In practice, you'd use a rotation matrix.
    # For simplicity, we'll assume the original_vector is already aligned and skip rotation.

    # Rotate random_vector_aligned back to the original coordinate system
    # This step is skipped in this simplified example. In practice, you'd apply the inverse rotation here.

    return random_vector_aligned


def visualize_cone_with_vector(original_vector, random_vector, cone_angle_deg):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    print(original_vector)
    # Plot the original vector
    ax.quiver(
        0,
        0,
        0,
        original_vector[0],
        original_vector[1],
        original_vector[2],
        color="blue",
        label="Original Vector",
    )

    # Plot the random vector
    ax.quiver(
        0,
        0,
        0,
        random_vector[0],
        random_vector[1],
        random_vector[2],
        color="red",
        label="Random Vector",
    )

    # Generate points for the cone's surface
    cone_length = 1  # Use 1 for simplicity, as we're dealing with unit vectors
    cone_radius = cone_length * np.tan(np.radians(cone_angle_deg))
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, cone_length, 100)
    u, v = np.meshgrid(u, v)
    x = cone_radius * v / cone_length * np.cos(u)
    y = cone_radius * v / cone_length * np.sin(u)
    z = v

    # Since we assumed the original vector is aligned with the z-axis, no rotation is applied to the cone
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.5, label="Cone Surface")

    # Setting the aspect ratio, labels, and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([0, 1])
    ax.view_init(elev=20, azim=30)
    plt.legend()

    plt.savefig("plot.pdf")


import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from numpy.linalg import norm


def generate_and_transform_cone_points(
    cone_length, cone_angle_deg, direction, position
):
    """
    Generates points for the cone's surface and transforms them to align with a given direction and position.

    Parameters:
    - cone_length: Length of the cone.
    - cone_angle_deg: Opening angle of the cone in degrees.
    - direction: The direction vector the cone should point towards.
    - position: The position vector where the cone's tip should be located.

    Returns:
    - Transformed x, y, z points of the cone's surface.
    """
    # Generate points for the cone's surface assuming it's aligned with the z-axis
    cone_radius = cone_length * np.tan(np.radians(cone_angle_deg))
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, cone_length, 100)
    u, v = np.meshgrid(u, v)
    x = cone_radius * v / cone_length * np.cos(u)
    y = cone_radius * v / cone_length * np.sin(u)
    z = v

    # Flatten the arrays to apply transformations
    points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

    # Step 2: Apply rotation to align the cone's axis with the given direction
    # Normalize the direction vector
    direction_normalized = direction / norm(direction)
    # Compute the rotation needed to align [0, 0, 1] with the direction vector
    rotation_axis = np.cross([0, 0, 1], direction_normalized)
    rotation_angle = np.arccos(np.dot([0, 0, 1], direction_normalized))
    # Generate the rotation matrix
    if norm(rotation_axis) != 0:  # Avoid division by zero for parallel vectors
        rotation = R.from_rotvec(rotation_axis / norm(rotation_axis) * rotation_angle)
        rotated_points = rotation.apply(points.T).T
    else:
        rotated_points = points  # No rotation needed if direction is parallel to z-axis

    # Step 3: Translate the cone to the given position
    translated_points = rotated_points + position[:, np.newaxis]

    return (
        translated_points[0].reshape(u.shape),
        translated_points[1].reshape(u.shape),
        translated_points[2].reshape(u.shape),
    )


def generate_and_transform_cone_for_vector(
    base_position, direction, cone_length, cone_angle_deg
):
    """
    Generates and transforms cone points to align with a given vector.

    Parameters:
    - base_position: The position of the base of the cone (vector's tip).
    - direction: The direction vector the cone should point towards.
    - cone_length: Length of the cone (should be less than or equal to the vector's length for visual clarity).
    - cone_angle_deg: Opening angle of the cone in degrees.

    Returns:
    - Transformed x, y, z points of the cone's surface for plotting.
    """
    # Direction normalization and cone tip calculation
    direction_normalized = direction / norm(direction)
    cone_tip_position = base_position - direction_normalized * cone_length

    # Generate cone points
    x_transformed, y_transformed, z_transformed = generate_and_transform_cone_points(
        cone_length, cone_angle_deg, direction_normalized, cone_tip_position
    )

    return x_transformed, y_transformed, z_transformed


def generate_and_transform_cone_at_vector_start(
    start_position, direction, cone_length, cone_angle_deg
):
    """
    Generates and transforms cone points to align with a given vector, starting the cone at the vector's start.

    Parameters:
    - start_position: The starting position of the vector (also the cone's tip).
    - direction: The direction vector the cone should point towards.
    - cone_length: Length of the cone.
    - cone_angle_deg: Opening angle of the cone in degrees.

    Returns:
    - Transformed x, y, z points of the cone's surface for plotting.
    """
    # Normalize the direction vector to ensure consistent behavior
    direction_normalized = direction / norm(direction)

    # Generate cone points
    x_transformed, y_transformed, z_transformed = generate_and_transform_cone_points(
        cone_length, cone_angle_deg, direction_normalized, start_position
    )

    return x_transformed, y_transformed, z_transformed


from typing import List
from trajectories import Waypoint, DiscretizedTrajectory
import matplotlib.cm as cm


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


cmaps = [cm.bone, cm.pink, cm.summer]


def visualise_vector_cones(positions, cone_angle_deg, traj):

    startpos = positions[0]
    firstdir = positions[1] - startpos
    # Plotting adjustments
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    # draw initial arrow
    ax.quiver(
        startpos[0],
        startpos[1],
        startpos[2],
        firstdir[0],
        firstdir[1],
        firstdir[2],
        color="black",
        length=norm(firstdir),
        arrow_length_ratio=0.1,
    )

    # draw rest
    for i in range(1, len(positions) - 1):
        cur_pos = positions[i]
        cur_dir = positions[i + 1] - cur_pos

        vector_magnitude = norm(cur_dir)

        cone_length = (
            0.25 * vector_magnitude
        )  # Make the cone's length proportional to the vector's magnitude

        # Correcting cone generation to start at the vector's start
        cone_dir = cur_pos - positions[i - 1]
        x_transformed, y_transformed, z_transformed = (
            generate_and_transform_cone_at_vector_start(
                cur_pos,  # Cone's tip at the vector's start
                cone_dir,
                cone_length,
                cone_angle_deg,
            )
        )

        # Plotting the cone
        ax.plot_surface(
            x_transformed, y_transformed, z_transformed, color="cyan", alpha=0.5
        )

        # Plotting the vector extending from the cone's tip
        ax.quiver(
            cur_pos[0],
            cur_pos[1],
            cur_pos[2],
            cur_dir[0],
            cur_dir[1],
            cur_dir[2],
            color="black",
            length=vector_magnitude,
            arrow_length_ratio=0.1,
        )

    # traj_np = discr_traj_to_ndarray(traj)
    #
    # x = traj_np[0]
    # y = traj_np[1]
    # z = traj_np[2]
    #
    # time_sequence = np.linspace(0.25, 0.75, len(traj))[::-1]
    #
    # ax.scatter(x, y, z, c=cmaps[0](time_sequence), marker='.', s=1, label='Generated polynomial trajectory')

    # Plot the random vector
    # ax.quiver(firstpos[0], firstpos[1], firstpos[2], scndpos[0], scndpos[1], scndpos[2], color='red', label='Random Vector')

    # Since we assumed the original vector is aligned with the z-axis, no rotation is applied to the cone
    # ax.plot_wireframe(x, y, z, color='gray', alpha=0.5, label='Cone Surface')

    # Setting the aspect ratio, labels, and legend
    padding = 0.5

    xmin = min([pos[0] for pos in positions])
    xmax = max([pos[0] for pos in positions])
    ymin = min([pos[1] for pos in positions])
    ymax = max([pos[1] for pos in positions])
    zmin = min([pos[2] for pos in positions])
    zmax = max([pos[2] for pos in positions])

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_xlim([xmin - padding, xmax + padding])
    ax.set_ylim([ymin - padding, ymax + padding])
    ax.set_zlim([zmin - padding, zmax + padding])
    ax.view_init(elev=20, azim=30)
    plt.legend()
    plt.savefig("random_wp_gen.pdf")


def vis_polynomial(positions, cone_angle_deg, traj):

    startpos = positions[0]
    firstdir = positions[1] - startpos
    # Plotting adjustments
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    traj_np = discr_traj_to_ndarray(traj)

    x = traj_np[0]
    y = traj_np[1]
    z = traj_np[2]

    time_sequence = np.linspace(0.25, 0.75, len(traj))[::-1]

    ax.scatter(x, y, z, c=cmaps[0](time_sequence), marker=".", s=1)

    wps = [Waypoint(coordinate=pos, timestamp=i) for i, pos in enumerate(positions)]
    wps_np = wps_to_ndarray(wps)
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
        color="black",
        marker="x",
        s=100,
        label="Target waypoints",
    )

    # Plot the random vector
    # ax.quiver(firstpos[0], firstpos[1], firstpos[2], scndpos[0], scndpos[1], scndpos[2], color='red', label='Random Vector')

    # Since we assumed the original vector is aligned with the z-axis, no rotation is applied to the cone
    # ax.plot_wireframe(x, y, z, color='gray', alpha=0.5, label='Cone Surface')

    # Setting the aspect ratio, labels, and legend
    padding = 0.5

    xmin = min([pos[0] for pos in positions])
    xmax = max([pos[0] for pos in positions])
    ymin = min([pos[1] for pos in positions])
    ymax = max([pos[1] for pos in positions])
    zmin = min([pos[2] for pos in positions])
    zmax = max([pos[2] for pos in positions])

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_xlim([xmin - padding, xmax + padding])
    ax.set_ylim([ymin - padding, ymax + padding])
    ax.set_zlim([zmin - padding, zmax + padding])
    ax.view_init(elev=20, azim=30)
    import matplotlib.lines as mlines

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=cmaps[0]([0.5]),
            marker="o",
            linestyle="None",
            markersize=5,
            label="Generated polynomial trajectory",
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
    plt.savefig("minimum_snap_traj.pdf")


def dummyvis():
    # Example original vector and parameters
    original_vector = np.array(
        [0, 0, 1]
    )  # Assuming this is already aligned with z-axis for simplicity
    cone_angle_deg = 30  # Cone angle in degrees
    std_deviation = np.radians(
        5
    )  # Standard deviation in radians for the angle within the cone

    # Generate a random vector
    random_vector = generate_random_vector_in_cone(
        original_vector, cone_angle_deg, std_deviation
    )

    visualize_cone_with_vector(original_vector, random_vector, cone_angle_deg)


def rand_traj_vis():
    from trajectories import TrajectoryFactory

    std_dev = 30
    traj, ctrl_wps = TrajectoryFactory.gen_random_trajectory(
        start=np.array([0, 0, 1]),
        n_discr_level=1000,
        n_ctrl_points=5,
        std_dev_deg=std_dev,
        distance_between_ctrl_points=1,
        init_dir=np.array([0, 1, 0]),
        return_ctrl_points=True,
    )
    positions = [wp.coordinate for wp in ctrl_wps]

    visualise_vector_cones(positions, std_dev, traj)
    vis_polynomial(positions, std_dev, traj)


if __name__ == "__main__":
    # Visualize the original vector, the random vector, and the cone
    rand_traj_vis()
    # dummyvis()
