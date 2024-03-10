"""Script learns an agent to follow a target trajectory.
"""
import sys 
sys.path.append("../..")
import argparse
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from aviaries.factories.position_controller_factory_test import PositionControllerFactoryTest
from agents.utils.configuration import Configuration
from agents.train_policy import run_train
from agents.test_policy import run_test
import numpy as np
from trajectories import TrajectoryFactory
# defaults for command line arguments
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_GUI = True
DEFAULT_TIMESTEPS = 5e6
DEFAULT_ACTION_TYPE = ActionType.RPM
DEFAULT_TRAIN = False
DEFAULT_TEST = True

# more configurations
DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'


def run(output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        timesteps=DEFAULT_TIMESTEPS,
        action_type: str = DEFAULT_ACTION_TYPE,
        train: bool = DEFAULT_TRAIN,
        test: bool = DEFAULT_TEST
    ):

    base_traj = TrajectoryFactory.get_linear_square_traj_discretized(
        n_discretization_level=4
    )
    wps = [wp for wp in base_traj]
    t_traj = TrajectoryFactory.get_pol_discretized_trajectory(
        t_waypoints=wps,
        n_points_discretization_level=4
    )
    config = Configuration(
        action_type=action_type,
        initial_xyzs=np.array([[0, 0, 1]]),
        output_path_location=output_folder,
        n_timesteps=timesteps,
        local=False,
        t_traj=t_traj
    )

    env_factory = PositionControllerFactoryTest(
        config=config,
        output_folder=output_folder,
        observation_type=DEFAULT_OBS,
        use_gui_for_test_env=gui,
        n_env_training=20,
        seed=0,
    )

    if test:
        run_test(config=config,
                 env_factory=env_factory)


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--timesteps',          default=DEFAULT_TIMESTEPS,     type=int,           help='number of train timesteps before stopping', metavar='')
    parser.add_argument('--train',          default=DEFAULT_TRAIN,     type=str2bool,           help='Whether to train (default: True)', metavar='')
    parser.add_argument('--test',          default=DEFAULT_TEST,     type=str2bool,           help='Whether to test (default: True)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))