import subprocess
import itertools
from concurrent.futures import ProcessPoolExecutor

# Define your grid parameters
param_grid = {
    'waypoint_buffer_size': [2, 3],
    'k_p' : [0.5, 1],
    'k_wp' : [4, 8],
    'k_s' : [0.5, 1],
    'max_reward_distance': [0.1, 0.2],
    'waypoint_dist_tol': [0.08, 0.12],
}

# Generate all combinations of parameters
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Function to run your program with given parameters
def run_program(combo):
    args = ['python', 'agents/C_trajectory_following.py']  # Base command to run your program
    args += [f"--{k}={v}" for k, v in combo.items()]  # Add parameters
    print(f"Running with arguments: {args[2:]}")  # Print current parameter set
    subprocess.run(args)

print(len(combinations))

# Run subprocesses in parallel
with ProcessPoolExecutor() as executor:
    executor.map(run_program, combinations)