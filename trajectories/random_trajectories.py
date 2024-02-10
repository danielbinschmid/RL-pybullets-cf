import numpy as np
from typing import Tuple
from .waypoint import Waypoint
from .discretized_trajectory import DiscretizedTrajectory
from typing import Optional

LOOP_COUNT_LIMIT=10000
Z_AXIS = np.array([0, 0, 1])

def normalise_vec(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vec / norm

def rotation_mat_to_z_axis(v: np.ndarray):
    v_unit = normalise_vec(v)

    axis_rotation = np.cross(v_unit, Z_AXIS)
    axis_normalized = normalise_vec(axis_rotation)
    # Angle of rotation (using the dot product)
    angle = np.arccos(np.dot(v_unit, axis_normalized))

    # Using the Rodrigues' rotation formula to compute the rotation matrix
    K = np.array([[0, -axis_normalized[2], axis_normalized[1]],
                  [axis_normalized[2], 0, -axis_normalized[0]],
                  [-axis_normalized[1], axis_normalized[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return rotation_matrix


class ControlPointSampler:
    def __init__(self, std_dev_deg: float, distance: float) -> None:
        """
        std_dev_deg: Standard deviation to the last direction vector in degrees \in [0, 360].
        distance: Distance from last waypoint to new sampled waypoint. 
        """
        assert(std_dev_deg >= 0 and std_dev_deg <= 360)
        assert(distance > 0)

        self.std_dev_deg = std_dev_deg
        self.distance = distance

    def sample_nxt_wp(self, wp: np.ndarray, current_direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO: better handling that waypoint stays above ground (z bigger than 0). Currently,
        sample until waypoint above ground is found.
        """
        # check for correct input shape
        assert(len(wp) == 3 and len(wp.shape) == 1)
        assert(len(current_direction) == 3 and len(current_direction.shape) == 1)

        # rotation to up-vector (0,0,1) in xyz system.
        rot_mat = rotation_mat_to_z_axis(current_direction)
        inv_rot_mat = rot_mat.T

        # TODO improve handling of infeasible waypoint below ground
        while_loop_count = 0
        new_wp = np.array([0,0,-1])
        while new_wp[2] < 0:
            rand_dir = self._sample_random_vec()
            projected_rand_dir = normalise_vec(np.dot(inv_rot_mat, rand_dir))
            new_wp = wp + self.distance * projected_rand_dir
            while_loop_count += 1

            if while_loop_count > LOOP_COUNT_LIMIT:
                raise SystemError("System got stuck in generating random trajectory.")

        return new_wp, projected_rand_dir

    def _sample_random_vec(self) -> np.ndarray:
        std_dev_rad = np.deg2rad(self.std_dev_deg)

        # assure that the standard deviation is given in radiance
        assert(std_dev_rad >= 0 and std_dev_rad <= np.pi * 2)
        
        # sample difference from z-axis from a gaussian distribution in radiance
        theta_rad = np.random.normal(0, std_dev_rad, 1)[0]
        # sample a random azimuthal angle
        phi_rad = np.random.uniform(0, 2 * np.pi)

        # position from angles
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)

        return np.array([x, y, z])
    

def sample_random_unit_vector():
    v = np.random.randn(3)
    unit_vector = v / np.linalg.norm(v)
    return unit_vector

def sample_random_ctrl_points(start_wp: np.ndarray,std_dev_deg: float=90, n_ctrl_points: int=10, distance_between_ctrl_point: float= 1, init_dir: Optional[np.ndarray]=None):
    
    if init_dir is None:
        cur_dir = sample_random_unit_vector()
    else:
        cur_dir = normalise_vec(init_dir)
    cur_wp = start_wp

    wp_sampler = ControlPointSampler(
        std_dev_deg=std_dev_deg,
        distance=distance_between_ctrl_point
    )
    wps = [
        Waypoint(
            coordinate=start_wp,
            timestamp=0
        )
    ]
    for i in range(n_ctrl_points):
        new_wp_coordinate, new_dir = wp_sampler.sample_nxt_wp(
            wp=cur_wp,
            current_direction=cur_dir
        )
        new_wp = Waypoint(
                coordinate=new_wp_coordinate,
                timestamp=i+1
        )
        wps.append(
            new_wp
        )
        cur_wp = new_wp_coordinate
        cur_dir = new_dir
    return wps
    
if __name__ == "__main__":
    cur_point = np.array([1,0,0])

    cur_dir = np.array([1, 0, 0])

    sampler = ControlPointSampler(
        std_dev_deg=90,
        distance=1
    )

    new_wp, new_dir = sampler.sample_nxt_wp(cur_point, cur_dir)
