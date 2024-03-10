#!/bin/bash
TRACKS="./eval-v0_n-ctrl-points-10_n-tracks-200_2024-02-12_15:01:53_bdee17ed-07ad-4e26-8e68-fa0a5ca1318c"

python pid.py --discr_level 26 --eval_set $TRACKS
# python C_trajectory_following.py --output_folder "../../checkpointed_models/" --discr_level 26 --eval_set $TRACKS
