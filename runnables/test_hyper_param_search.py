import os 
import subprocess

# Function to run your program with given parameters
def run_program(combo):
    args = ['python', 'agents/C_trajectory_following.py']  # Base command to run your program
    args += [f"--{k}={v}" for k, v in combo.items()]  # Add parameters
    print(f"Running with arguments: {args[2:]}")  # Print current parameter set
    subprocess.run(args)

wp_buffer_size=3
os.chdir("..")
checkpoints_base_folder = "/shared/d/master/ws23/UAV-lab/git_repos/RL-pybullets-cf/agents/experiment_results/exp-C/grid_search_results/buffer3_kp1"

experiments  = os.listdir(checkpoints_base_folder)
experiment_folders = [os.path.join(checkpoints_base_folder, experiment) for experiment in experiments] 

for experiment_folder in experiment_folders:
    args = ['python', 'agents/C_trajectory_following.py', ]
    args+= [f'--output_folder={experiment_folder}']
    args+= [f'--waypoint_buffer_size={wp_buffer_size}']
    args+= ['--train=False']
    args+= ['--test=True']
    print(f"Running with arguments: {args[2:]}")  # Print current parameter set
    subprocess.run(args)
