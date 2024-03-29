"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""

import sys

sys.path.append("../..")

import pybullet as p
import time
import argparse
import numpy as np
from trajectories import TrajectoryFactory, DiscretizedTrajectory

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from runnables.utils.gen_eval_tracks import load_eval_tracks
from typing import List, Dict
import json
from tqdm import tqdm

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 50
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False
DEFAULT_DISCR_LEVEL = 10
DEFAULT_EVAL_SET = "./eval-v0_n-ctrl-points-3_n-tracks-20_2024-02-11_22:18:28_46929077-0248-4c6e-b2b1-da2afb13b2e2"

from runnables.utils.utils import compute_metrics_single


def save_benchmark(benchmarks: Dict[str, float], file_path: str):
    with open(file_path, "w") as file:
        json.dump(benchmarks, file)


def run(
    drone=DEFAULT_DRONES,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VISION,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    obstacles=DEFAULT_OBSTACLES,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB,
    discr_level=DEFAULT_DISCR_LEVEL,
    eval_set=DEFAULT_EVAL_SET,
):
    #### Initialize the simulation #############################

    #### Initialize a circular trajectory ######################
    use_gui = True
    n_discr_level = discr_level

    eval_set_folder = eval_set
    tracks = load_eval_tracks(eval_set_folder, discr_level=n_discr_level)
    mean_devs = []
    max_devs = []
    successes = []
    times = []
    for track in tqdm(tracks):
        current_step = 0
        INIT_RPYS = np.array([[0.0, 0.0, 0.0]])

        TARGET_TRAJECTORY, init_wp = track, np.array([track[0].coordinate])

        #### Create the environment ################################
        env = CtrlAviary(
            drone_model=drone,
            num_drones=1,
            initial_xyzs=init_wp,
            initial_rpys=INIT_RPYS,
            physics=physics,
            neighbourhood_radius=10,
            pyb_freq=simulation_freq_hz,
            ctrl_freq=control_freq_hz,
            gui=use_gui,
            record=record_video,
            obstacles=False,
            user_debug_gui=user_debug_gui,
            log_positions=True,
        )

        #### Obtain the PyBullet Client ID from the environment ####
        PYB_CLIENT = env.getPyBulletClient()

        #### Initialize the logger
        logger = Logger(
            logging_freq_hz=control_freq_hz,
            num_drones=num_drones,
            output_folder=output_folder,
            colab=colab,
        )

        #### Initialize the controller
        drone = DroneModel.CF2X
        ctrl = DSLPIDControl(drone_model=drone)

        #### Run the simulation
        action = np.zeros((num_drones, 4))
        START = time.time()

        for point in TARGET_TRAJECTORY:
            sphere_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.03,
                rgbaColor=[0, 1, 0, 0.6],
                physicsClientId=env.CLIENT,
            )
            target = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=sphere_visual,
                basePosition=point.coordinate,
                useMaximalCoordinates=False,
                physicsClientId=env.CLIENT,
            )
            p.changeVisualShape(
                target, -1, rgbaColor=[0.9, 0.3, 0.3, 0.6], physicsClientId=env.CLIENT
            )
        for i in range(0, int(duration_sec * env.CTRL_FREQ)):
            #### Make it rain rubber ducks #############################
            # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

            #### Step the simulation ###################################
            obs, reward, terminated, truncated, info = env.step(action)
            target_position = TARGET_TRAJECTORY[current_step].coordinate
            #### Compute control for the current way point #############
            action, _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[0],
                target_pos=np.hstack([target_position]),
                # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                target_rpy=INIT_RPYS[0],
            )

            action = np.array([action])

            #### Go to the next way point and loop #####################
            position = obs[0][0:3]

            distance = np.linalg.norm(target_position - position)
            velocity = np.linalg.norm(obs[0][10:13])
            if distance < 0.2 and velocity < 0.05:
                if current_step == len(TARGET_TRAJECTORY) - 1 and velocity < 0.05:
                    # env.render()
                    all_pos = env.pos_logger.load_all()
                    t = env.step_counter * env.PYB_TIMESTEP
                    mean_dev, max_dev = compute_metrics_single(all_pos, track)
                    mean_devs.append(mean_dev)
                    max_devs.append(max_dev)
                    times.append(t)
                    successes.append(True)
                    break
                current_step = (current_step + 1) % len(TARGET_TRAJECTORY)

            ##### Log the simulation ####################################

            #### Printout
            # env.render()

            #### Sync the simulation
            if gui and use_gui:
                sync(i, START, env.CTRL_TIMESTEP)
                i += 1

            if i == int(duration_sec * env.CTRL_FREQ) - 1:
                successes.append(False)
        #### Close the environment
        env.close()

    #### Save the simulation results ###########################
    logger.save()

    print(f"N DISCR LEVEL: {n_discr_level}")
    print("COMPLETION TIME MEAN:", np.mean(times))
    print("SUCCESS RATE:", np.mean(successes))
    print("AVERAGE DEVIATION: ", np.mean(mean_devs))
    print("MAXIMUM DEVIATION:", np.mean(max_devs))

    save_benchmark(
        {
            "success_rate": np.mean(successes),
            "avg mean dev": np.mean(mean_devs),
            "avg max dev": np.mean(max_devs),
            "avt time": np.mean(times),
        },
        f"pid_{discr_level}.json",
    )
    #############
    # if plot:
    #     logger.plot()


if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Helix flight script using CtrlAviary and DSLPIDControl"
    )
    parser.add_argument(
        "--drone",
        default=DEFAULT_DRONES,
        type=DroneModel,
        help="Drone model (default: CF2X)",
        metavar="",
        choices=DroneModel,
    )
    parser.add_argument(
        "--num_drones",
        default=DEFAULT_NUM_DRONES,
        type=int,
        help="Number of drones (default: 3)",
        metavar="",
    )
    parser.add_argument(
        "--physics",
        default=DEFAULT_PHYSICS,
        type=Physics,
        help="Physics updates (default: PYB)",
        metavar="",
        choices=Physics,
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VISION,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--plot",
        default=DEFAULT_PLOT,
        type=str2bool,
        help="Whether to plot the simulation results (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--user_debug_gui",
        default=DEFAULT_USER_DEBUG_GUI,
        type=str2bool,
        help="Whether to add debug lines and parameters to the GUI (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--obstacles",
        default=DEFAULT_OBSTACLES,
        type=str2bool,
        help="Whether to add obstacles to the environment (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--simulation_freq_hz",
        default=DEFAULT_SIMULATION_FREQ_HZ,
        type=int,
        help="Simulation frequency in Hz (default: 240)",
        metavar="",
    )
    parser.add_argument(
        "--control_freq_hz",
        default=DEFAULT_CONTROL_FREQ_HZ,
        type=int,
        help="Control frequency in Hz (default: 48)",
        metavar="",
    )
    parser.add_argument(
        "--duration_sec",
        default=DEFAULT_DURATION_SEC,
        type=int,
        help="Duration of the simulation in seconds (default: 5)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    parser.add_argument(
        "--discr_level",
        default=DEFAULT_DISCR_LEVEL,
        type=int,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    parser.add_argument(
        "--eval_set",
        default=DEFAULT_EVAL_SET,
        type=str,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    ARGS = parser.parse_args()

    run(**vars(ARGS))
