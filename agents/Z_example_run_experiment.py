"""Script learns an agent to follow a target trajectory.
"""

import argparse
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType
from factories.simple_follower_factory import EnvFactorySimpleFollowerAviary
from agents.utils.configuration import Configuration
from train_policy import run_train
from test_policy import run_test
from agents.utils.configuration import parse_config

# defaults for command line arguments
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_GUI = True
DEFAULT_TIMESTEPS = 3e5
DEFAULT_ACTION_TYPE = 'rpm' # 'rpm', 'one_d_rpm', 'attitude'
DEFAULT_TRAIN = False 
DEFAULT_TEST = True

# more configurations
DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'


def run(output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        timesteps=DEFAULT_TIMESTEPS,
        action_type: str='rpm',
        train: bool=DEFAULT_TRAIN,
        test: bool=DEFAULT_TEST
    ):

    config: Configuration = parse_config(
        t_waypoint=[0, 0.5, 0.5],
        initial_waypoint=[0, 0, 0.1],
        action_type="rpm",
        output_folder=output_folder,
        n_timesteps=timesteps,
        local=False
    )

    env_factory = EnvFactorySimpleFollowerAviary(
        config=config,
        output_folder=output_folder,
        observation_type=DEFAULT_OBS,
        use_gui_for_test_env=gui,
        n_env_training=20,
        seed=0,
    )

    if train:
        run_train(config=config,
                  env_factory=env_factory)

    if test:
        run_test(config=config,
                 env_factory=env_factory)


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--timesteps',          default=DEFAULT_TIMESTEPS,     type=int,           help='number of train timesteps before stopping', metavar='')
    parser.add_argument('--action_type',          default=DEFAULT_TIMESTEPS,     type=str,           help='Either "one_d_rpm", "rpm" or "attitude"', metavar='')
    parser.add_argument('--train',          default=DEFAULT_TRAIN,     type=str2bool,           help='Whether to train (default: True)', metavar='')
    parser.add_argument('--test',          default=DEFAULT_TEST,     type=str2bool,           help='Whether to test (default: True)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
