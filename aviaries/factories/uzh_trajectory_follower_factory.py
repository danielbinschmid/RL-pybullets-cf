import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import DiscretizedTrajectory
from stable_baselines3.common.vec_env import VecEnv
from aviaries.configuration import Configuration
from aviaries.UZHAviary import UZHAviary
from .base_factory import BaseFactory


class TrajectoryFollowerAviaryFactory(BaseFactory):
    action_type: ActionType
    observation_type: ObservationType
    t_traj: DiscretizedTrajectory
    n_env_training: int
    initial_xyzs: np.ndarray
    seed: int
    use_gui_for_test_env: bool
    output_path_location: str
    episode_len_sec: int
    waypoint_buffer_size: int
    k_p: float
    k_wp: float
    k_s: float
    max_reward_distance: float
    waypoint_dist_tol: float

    def __init__(
        self,
        config: Configuration,
        observation_type: ObservationType,
        use_gui_for_test_env: bool = True,
        n_env_training: int = 20,
        seed: int = 0,
        single_traj=False,
        eval_mode=False,
    ) -> None:
        super().__init__()
        self.eval_mode = eval_mode
        self.single_traj = single_traj
        self.observation_type = observation_type
        self.n_env_training = n_env_training
        self.seed = seed
        self.use_gui_for_test_env = use_gui_for_test_env
        self.set_config(config)

    def set_config(self, config: Configuration):
        initial_xyzs = config.initial_xyzs
        action_type = config.action_type
        t_traj = config.t_traj
        self.initial_xyzs = initial_xyzs
        self.action_type = action_type
        self.t_traj = t_traj
        self.episode_len_sec = config.episode_len_sec
        self.waypoint_buffer_size = config.waypoint_buffer_size
        self.k_p = config.k_p
        self.k_wp = config.k_wp
        self.k_s = config.k_s
        self.max_reward_distance = config.max_reward_distance
        self.waypoint_dist_tol = config.waypoint_dist_tol

    def get_train_env(self) -> VecEnv:
        train_env = make_vec_env(
            UZHAviary,
            env_kwargs=dict(
                target_trajectory=self.t_traj,
                initial_xyzs=self.initial_xyzs,
                obs=self.observation_type,
                act=self.action_type,
                episode_len_sec=self.episode_len_sec,
                waypoint_buffer_size=self.waypoint_buffer_size,
                k_p=self.k_p,
                k_wp=self.k_wp,
                k_s=self.k_s,
                max_reward_distance=self.max_reward_distance,
                waypoint_dist_tol=self.waypoint_dist_tol,
                one_traj=self.single_traj,
            ),
            n_envs=self.n_env_training,
            seed=self.seed,
        )
        return train_env

    def get_eval_env(self):
        eval_env = UZHAviary(
            target_trajectory=self.t_traj,
            initial_xyzs=self.initial_xyzs,
            obs=self.observation_type,
            act=self.action_type,
            episode_len_sec=self.episode_len_sec,
            waypoint_buffer_size=self.waypoint_buffer_size,
            k_p=self.k_p,
            k_wp=self.k_wp,
            k_s=self.k_s,
            max_reward_distance=self.max_reward_distance,
            waypoint_dist_tol=self.waypoint_dist_tol,
            one_traj=self.single_traj,
            eval_mode=self.eval_mode,
            log_positions=True if self.eval_mode else False,
        )
        return eval_env

    def get_test_env_gui(self):
        test_env = UZHAviary(
            target_trajectory=self.t_traj,
            initial_xyzs=self.initial_xyzs,
            gui=self.use_gui_for_test_env,
            obs=self.observation_type,
            act=self.action_type,
            record=False,
            episode_len_sec=self.episode_len_sec,
            waypoint_buffer_size=self.waypoint_buffer_size,
            k_p=self.k_p,
            k_wp=self.k_wp,
            k_s=self.k_s,
            max_reward_distance=self.max_reward_distance,
            waypoint_dist_tol=self.waypoint_dist_tol,
            one_traj=self.single_traj,
            eval_mode=self.eval_mode,
            log_positions=True if self.eval_mode else False,
        )
        return test_env

    def get_test_env_no_gui(self):
        test_env_nogui = UZHAviary(
            target_trajectory=self.t_traj,
            initial_xyzs=self.initial_xyzs,
            obs=self.observation_type,
            act=self.action_type,
            episode_len_sec=self.episode_len_sec,
            waypoint_buffer_size=self.waypoint_buffer_size,
            k_p=self.k_p,
            k_wp=self.k_wp,
            k_s=self.k_s,
            max_reward_distance=self.max_reward_distance,
            waypoint_dist_tol=self.waypoint_dist_tol,
            one_traj=self.single_traj,
            eval_mode=self.eval_mode,
            log_positions=True if self.eval_mode else False,
        )
        return test_env_nogui
