#!/bin/bash

for level in {14,17,20,23,26}
do
  # python C_trajectory_following.py --output_folder "../../checkpointed_models/" --eval_set "./eval-v0_n-ctrl-points-10_n-tracks-200_2024-02-12_15:01:53_bdee17ed-07ad-4e26-8e68-fa0a5ca1318c" --discr_level $level
  python pid.py --discr_level $level --eval_set "./eval-v0_n-ctrl-points-10_n-tracks-200_2024-02-12_15:01:53_bdee17ed-07ad-4e26-8e68-fa0a5ca1318c"

done
