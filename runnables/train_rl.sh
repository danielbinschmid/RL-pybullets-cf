#!/bin/bash

python ../agents/C_trajectory_following.py \
    --gui "True" \
    --output_folder "../checkpointed_models" \
    --timesteps 2500000 \
    --train "False" \
    --test "True" \
    --n_envs "5" \
    --episode_len_sec "20" \
    --waypoint_buffer_size "2" \
    --k_p "5" \
    --k_wp "8" \
    --k_s "0.05" \
    --max_reward_distance "0.0" \
    --waypoint_dist_tol "0.05"
