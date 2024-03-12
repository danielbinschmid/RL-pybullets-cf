import matplotlib.pyplot as plt
import os
import plotting
from action_types_experiments_data import iterations, rpm_task_up, one_d_rpm_task_up, attitude_task_up, iterations_2
import matplotlib.ticker as ticker
plotting.prepare_for_latex()


# Sample data
iterations = iterations
# Case 1
mean_reward_progression_1 = rpm_task_up
# Case 2
mean_reward_progression_2 = one_d_rpm_task_up

# Case 3
mean_reward_progression_3 = attitude_task_up 

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting Mean Reward Progression for each case with different line styles
ax1.plot(iterations, mean_reward_progression_1, 'bo-', label='Reward ActionType RPM') # solid
ax1.plot(iterations, mean_reward_progression_2, 'ro--', label='Reward ActionType ONE-D-RPM') # dashed
ax1.plot(iterations_2, mean_reward_progression_3, 'go-.', label='Reward ActionType Attitude') # dash-dot

# Labels, title and grid
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Mean Reward', color='black')
plt.title('Mean Reward Progression over Iterations')

ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0e}'.format(x)))

# Only show every 10th iteration
ax1.set_xticks(iterations[::10])

# Legend
ax1.legend(loc='lower right')

plt.grid(True)

res_fname = "action_type_experiment_2.pdf"
if os.path.exists(res_fname):
    os.remove(res_fname)
plt.savefig(res_fname)
