#!/bin/bash

# change directory to agents
cd ..
cd agents

# run experiments
# python ./experiment_action_types.py --timesteps "2000000" --gui "False" --output_folder "rpm_task_up" --action_type "rpm"
# python ./experiment_action_types.py --timesteps "2000000" --gui "False" --output_folder "one_d_rpm_task_up" --action_type "one_d_rpm"
python ./experiment_action_types.py --timesteps "2000000" --gui "False" --output_folder "attitude_task_up" --action_type "attitude"
