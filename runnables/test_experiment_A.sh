#!/bin/bash

# change directory to agents
cd ..
cd agents

timesteps="2000000"
gui="True"
test="True"
train="False"
base_output_folder="exp-A"
total_runs=1

declare -a action_types=("attitude")
declare -a modes=("DIAGONAL_UP, DIAGONAL_DOWN, UP, DOWN")

for action_type in "${action_types[@]}"; do
    for mode in "${modes[@]}"; do
        for (( run=1; run<=total_runs; run++ )); do
            output_folder="${base_output_folder}_${action_type}_${mode}_run${run}"
            python ./run_experiment_A.py --timesteps "$timesteps" --gui "$gui" --output_folder "$output_folder" --test "$test" --action_type "$action_type" --train "$train" --mode "$mode" 
        done
    done
done

wait # Wait for all background processes to finish
