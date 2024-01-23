"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid_follow_trajectory.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import sys 
sys.path.append("..")


import time
import argparse
import numpy as np
from trajectories import TrajectoryFactory, Waypoint

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

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
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

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
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[0., 0., 0.5]])
    INIT_RPYS = np.array([[0., 0., 0.]])

    #### Initialize a polynomial trajectory ####################
    # initial_xyzs = np.array([[0.,     0.,     0.5]])

    # example trajectory
    t_wps = TrajectoryFactory.waypoints_from_numpy(
        np.asarray([
            [0, 0, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ])
    )
    target_trajectory = TrajectoryFactory.get_discr_from_wps(t_wps)

    for wp in range(len(target_trajectory)):
        print(target_trajectory[wp].coordinate)
    n_waypoints = len(target_trajectory)
    cur_waypoint_idx = 0

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
        num_drones=1,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui
    )

    #### Initialize the logger
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab
    )

    #### Initialize the controller
    drone = DroneModel.CF2X
    ctrl = DSLPIDControl(drone_model=drone)

    #### Run the simulation
    action = np.zeros((num_drones,4))
    START = time.time()

    final_coordinate = target_trajectory[len(target_trajectory) - 1].coordinate

    done = False
    # print(final_coordinate)
    for i in range(int(1e5)):

        #### Step the simulation ###################################
        obs, _, _, _, _ = env.step(action)
        target_position = target_trajectory[cur_waypoint_idx].coordinate

        #### Compute control for the current way point #############
        action, _, _ = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=np.hstack([target_position]),
            target_rpy=INIT_RPYS[0]
        )

        action = np.array([action])

        #### Go to the next way point and loop #####################
        position = obs[0][0:3]
        distance2final = np.linalg.norm(final_coordinate - position)
        distance = np.linalg.norm(target_position - position)
        velocity = np.linalg.norm(obs[0][10:13])
        
        if distance2final < 0.1 and cur_waypoint_idx >= len(target_trajectory) - 5:
            done = True
            
        if distance < 0.1 and not done:
            cur_waypoint_idx = (cur_waypoint_idx + 1) % n_waypoints

        
        # if distance2final < 0.1:
        #     break
        ##### Log the simulation ####################################

        #### Printout
        env.render()

        #### Sync the simulation
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
            i += 1

    #### Close the environment
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
