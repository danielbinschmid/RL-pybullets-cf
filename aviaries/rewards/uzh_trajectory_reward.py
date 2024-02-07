import numpy as np
import pybullet as p

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class RewardDict: 
    def __init__(self, r_t: float=0, r_p: float=0, r_wp:float=0, r_s: float=0, r_w: float=0) -> None:
        self.r_t = r_t
        self.r_p = r_p 
        self.r_wp = r_wp 
        self.r_s = r_s 
        self.r_w = r_w
    
    def __str__(self) -> str:
        return f'r_t: {self.r_t:.3f}; r_p: {self.r_p:.3f}; r_wp: {self.r_wp:.3f}; r_s: {self.r_s:.3f}; r_w: {self.r_w}'

    def sum(self):
        return self.r_t + self.r_p + self.r_wp + self.r_s + self.r_w
    
class Rewards:
    cur_reward: RewardDict

    def __init__(self, 
                 trajectory: np.ndarray,
                 k_p: float=1.5,
                 k_wp: float=3,
                 k_s: float=0.07,
                 k_w: float=0,
                 max_reward_distance: float=0.2,
                 dist_tol: float=0.12) -> None:
        self.trajectory = trajectory

        # intermediates
        self.p1 = self.trajectory[:-1]
        self.p2 = self.trajectory[1:]
        self.diffs = self.p2 - self.p1
        self.distances = np.linalg.norm(self.p1 - self.p2, axis=1)
        self.reached_distance = 0
        self.current_projection_distance = 0
        self.current_projection = self.trajectory[0]

        # weights for reward
        self.k_p = k_p
        self.k_wp = k_wp
        self.k_s = k_s 
        self.k_w = k_w
        print(f'k_p: {k_p}; k_wp: {k_wp}; k_s: {k_s}; k_w: {k_w}')

        self.wp_rewards = np.zeros(len(self.trajectory))
        self.max_reward_distance = max_reward_distance
        
        self.dist_tol = dist_tol
        self.cur_reward = RewardDict()
        self.cur_wp_idx = 0
    def reset(self, trajectory):
        self.cur_reward = RewardDict()
        self.trajectory = trajectory
        self.wp_rewards = np.zeros(len(self.trajectory))

    def get_projections(self, position: np.ndarray):
        """
        Returns closest points from the drone to each trajectory segment
        """
        shifted_position = position - self.p1
        dots = np.einsum('ij,ij->i', shifted_position, self.diffs)
        norm = np.linalg.norm(self.diffs, axis=1) ** 2
        coefs = dots / (norm + 1e-9)
        coefs = np.clip(coefs, 0, 1)
        projections = coefs[:, np.newaxis] * self.diffs + self.p1
        return projections
    
    def get_travelled_distance(self, position: np.ndarray):
        """
        current_projection: closest point on the trajectory to the drone
        current_projection_idx: index of the closest waypoint on the segmented trajectory to the drone that is next
        overall_distance_travelled: total distance travelled by the drone (on trajectory)
        """
        projections = self.get_projections(position)
        displacement_size = np.linalg.norm(projections-position, axis=1)
        closest_point_idx = np.argmin(displacement_size)

        current_projection = projections[closest_point_idx]
        self.current_projection = current_projection
        current_projection_idx = min(closest_point_idx + 1, len(self.trajectory) - 1)

        overall_distance_travelled = np.sum(self.distances[:closest_point_idx]) \
            + np.linalg.norm(projections[closest_point_idx] - self.p1[closest_point_idx])
        
       
        return current_projection, current_projection_idx, overall_distance_travelled
    
    def get_closest_waypoint(self, position: np.ndarray):
        """
        Find the closest waypoint to the drone (for waypoint reward)
        """
        distance = np.linalg.norm(self.trajectory[self.cur_wp_idx + 1] - position)
        return distance, self.cur_wp_idx + 1

    def weight_rewards(self, r_t, r_p, r_wp, r_s, r_w=0):
        self.cur_reward = RewardDict(
            r_t=r_t,
            r_p=self.k_p * r_p,
            r_wp=self.k_wp * r_wp,
            r_s=self.k_s * r_s,
            r_w=self.k_w * r_w
        )

        return self.cur_reward.sum()

    def compute_reward(self, drone_state: np.ndarray, reached_distance: np.ndarray, bodyrates=None):
        """
        r_t - reward for crashing
        r_p - reward for delta distance travelled
        r_s - reward for overall distance travelled
        r_wp - reward for reaching waypoint 
        TODO high body rates punishment
        """
        position = drone_state[:3]
        closest_waypoint_distance, closest_waypoint = self.get_closest_waypoint(position)
        projection_distance = np.linalg.norm(self.current_projection - position)

        r_t = -10 if (abs(drone_state[7]) > .4 or abs(drone_state[8]) > .4) else 0 # when its tilted 
        r_p = reached_distance - self.reached_distance
        r_s = reached_distance
        print("WPCUR", self.cur_wp_idx)
        # If we are passing waypoint for the first time, give reward for passing it and remember it
        if closest_waypoint_distance <= self.dist_tol and not self.wp_rewards[closest_waypoint]:
            self.wp_rewards[closest_waypoint] = 1
            r_wp = np.exp(-closest_waypoint_distance/self.dist_tol)
            self.cur_wp_idx = self.cur_wp_idx + 1 if self.cur_wp_idx < len(self.trajectory) - 2 else len(self.trajectory) - 2
            
        else:
            r_wp = 0

        if bodyrates is not None:
            r_w = - np.linalg.norm(bodyrates)
        else:
            r_w = 0

        # weighting by velocity
        velocity = drone_state[10:13] 
        velocity_norm = np.linalg.norm(velocity)
        min_vel = 0.2
        max_vel = 0.5
        s_vmax = (5**(max_vel - velocity_norm)) if velocity_norm > max_vel else 1
        s_min = (5**(velocity_norm - min_vel)) if velocity_norm < min_vel else 1
        s_gd = np.exp(self.max_reward_distance - projection_distance) if projection_distance > self.max_reward_distance else 1
        scale = s_vmax * s_min * s_gd
        # print(f'velocity size is {velocity_norm}')
        r = self.weight_rewards(r_t, scale*r_p, r_wp, scale*r_s, r_w)
        # print(f'max reward disance {self.max_reward_distance}')
        self.current_projection_distance = projection_distance
        self.reached_distance = reached_distance
        # print(f'velocity is {np.linalg.norm(velocity)}')
        # print(f'projection distance is {projection_distance}')
        # print(f'r is {r}')
        return r
    
    def __str__(self) -> str:
        return ""