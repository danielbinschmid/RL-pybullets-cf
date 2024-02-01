from aviaries.factories.base_factory import BaseFactory
from agents.utils.configuration import Configuration
import os
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.Logger import Logger
from stable_baselines3 import PPO
import time
import numpy as np 
from gym_pybullet_drones.utils.utils import sync


def test_simple_follower(local: bool, filename: str, test_env_nogui: BaseRLAviary, test_env: BaseRLAviary, output_folder: str):

    # load model    
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    # evaluate in test environment
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    # visualise in test environment
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=1,
                output_folder=output_folder,
                colab=False
                )
    
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12)
                    )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})

    test_env.close()

    logger.plot()


def run_test(config: Configuration, env_factory: BaseFactory):


    test_env = env_factory.get_test_env_gui()
    test_env_nogui = env_factory.get_test_env_no_gui()

    test_simple_follower(
        local=config.local,
        filename=config.output_path_location,
        test_env_nogui=test_env_nogui,
        test_env=test_env,
        output_folder=config.output_path_location
    )


