#!/bin/bash

# change directory to agents
cd ..
cd agents

timesteps="10000000"
gui="False"
test="False"
train="True"
base_output_folder="exp-A"
total_runs=1

declare -a action_types=("attitude" "rpm")
declare -a modes=("DIAGONAL_UP" "UP" "DOWN" "SIDEWAYS" "DIAGONAL_UP" "DIAGONAL_DOWN")

for action_type in "${action_types[@]}"; do
    for mode in "${modes[@]}"; do
        for (( run=1; run<=total_runs; run++ )); do
            output_folder="${base_output_folder}_${action_type}_${mode}_run${run}"
            python ./run_experiment_A.py --timesteps "$timesteps" --gui "$gui" --output_folder "$output_folder" --test "$test" --action_type "$action_type" --train "$train" --mode "$mode" 
        done
    done
done

wait # Wait for all background processes to finish
