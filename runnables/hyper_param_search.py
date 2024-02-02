import subprocess
import itertools

# Define your grid parameters
param_grid = {
    'waypoint_buffer_size': [3,4],
    'k_p' : [1, 2],
    'k_wp' : [3, 7],
    'k_s' : [0.1, 0.2],
    'max_reward_distance': [0.1, 0.2],
    'waypoint_dist_tol': [0.05, 0.12],
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