import matplotlib.pyplot as plt
import os
import plotting
from action_types_experiments_data import iterations_exp2, rpm_exp2, attitude_exp2
import matplotlib.ticker as ticker
plotting.prepare_for_latex()


# Sample data
iterations = iterations_exp2
# Case 1
mean_reward_progression_1 = rpm_exp2
# Case 2
mean_reward_progression_2 = attitude_exp2

# Case 3

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting Mean Reward Progression for each case with different line styles
ax1.plot(iterations, mean_reward_progression_1, 'bo-', label='Reward ActionType RPM') # solid
ax1.plot(iterations, mean_reward_progression_2, 'ro--', label='Reward ActionType Attitude') # dashed

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

res_fname = "action_type_experiment_tilted.pdf"
if os.path.exists(res_fname):
    os.remove(res_fname)
plt.savefig(res_fname)
