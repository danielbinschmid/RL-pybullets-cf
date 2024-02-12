#!/bin/bash

# python pid.py --discr_level 9 --eval_set "./eval-v0_n-ctrl-points-3_n-tracks-200_2024-02-12_12:21:14_3115f5a9-bd48-4b63-9194-5eda43e1329d"
python C_trajectory_following.py --output_folder "../../checkpointed_models/" --discr_level 10 --eval_set "./eval-v0_n-ctrl-points-3_n-tracks-200_2024-02-12_12:21:14_3115f5a9-bd48-4b63-9194-5eda43e1329d"