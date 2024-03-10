# Reinforcement Learning for Quadcopter Control

This repository is a fork of [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) and implements a reinforcement learning based control policy inspired by [Penicka et al. [1]](https://rpg.ifi.uzh.ch/docs/RAL_IROS22_Penicka.pdf). 

## Result

![RL Control Result](./media/rl_control.gif)

- The drone can follow arbitrary trajectories. 
- It is given two next target waypoints as observation. If the two target waypoints are close, it will reach the target slower. 
- The learned policy is well-suited to smoothly follow arbitrary trajectories and corresponds to the obtained result after _slow-phase training_ in [Penicka et al. [1]](https://rpg.ifi.uzh.ch/docs/RAL_IROS22_Penicka.pdf).

## Implemented New Features

- Reward implementation of RL policy proposed by [Penicka et al. [1]](https://rpg.ifi.uzh.ch/docs/RAL_IROS22_Penicka.pdf).
- Attitude control action type. In [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones), only motor-level control using PWM signals is implemented. This repository extends the original implementation and adds a wrapper for sending attitude commands (thrust and bodyrates).
- Random trajectory generation using polynomial minimum snap trajectory generation using [large_scale_traj_optimizer [2]](https://github.com/ZJU-FAST-Lab/large_scale_traj_optimizer) for training and test set generation. Implementation in [trajectories](./trajectories/) subfolder.
- Scripts for bechmarking the policy by computing basic benchmarks such as mean and max deviation from the target trajectory and time until completion.

## Setup


## Implementation Overview

## References 

- [1]: Penicka, Robert, et al. [*Learning minimum-time flight in cluttered environments.*](https://rpg.ifi.uzh.ch/docs/RAL_IROS22_Penicka.pdf) IEEE Robotics and Automation Letters 7.3 (2022): 7209-7216.
- [2]: Burke, Declan, Airlie Chapman, and Iman Shames. [*Generating minimum-snap quadrotor trajectories really fast.*](https://ieeexplore.ieee.org/abstract/document/9341794) 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020. [github](https://github.com/ZJU-FAST-Lab/large_scale_traj_optimizer)