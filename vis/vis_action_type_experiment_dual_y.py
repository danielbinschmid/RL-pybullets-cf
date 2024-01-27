import matplotlib.pyplot as plt

# Sample data
iterations = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000, 320000]

# Case 1
mean_reward_progression_1 = [100, 120, 150, 170, 180, 190, 210, 230, 250, 270, 280, 290, 300, 310, 320, 330]
mean_episode_length_1 = [70, 72, 75, 77, 80, 82, 85, 87, 90, 92, 95, 97, 100, 102, 105, 107]

# Case 2
mean_reward_progression_2 = [90, 110, 130, 140, 160, 180, 200, 210, 230, 240, 260, 270, 290, 300, 310, 320]
mean_episode_length_2 = [65, 67, 70, 73, 75, 78, 80, 83, 85, 88, 90, 93, 95, 98, 100, 103]

# Case 3
mean_reward_progression_3 = [110, 130, 150, 160, 170, 200, 220, 240, 260, 280, 290, 300, 310, 320, 330, 340]
mean_episode_length_3 = [75, 78, 80, 82, 85, 87, 90, 92, 95, 97, 100, 102, 105, 107, 110, 113]

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting Mean Reward Progression for each case with different line styles
ax1.plot(iterations, mean_reward_progression_1, 'bo-', label='Reward Case 1') # solid
ax1.plot(iterations, mean_reward_progression_2, 'ro--', label='Reward Case 2') # dashed
ax1.plot(iterations, mean_reward_progression_3, 'go-.', label='Reward Case 3') # dash-dot

# Create a second y-axis for Mean Episode Length
ax2 = ax1.twinx()
ax2.plot(iterations, mean_episode_length_1, 'b*-', label='Episode Length Case 1') # solid
ax2.plot(iterations, mean_episode_length_2, 'r*--', label='Episode Length Case 2') # dashed
ax2.plot(iterations, mean_episode_length_3, 'g*-.', label='Episode Length Case 3') # dash-dot

# Labels, title and grid
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Mean Reward', color='black')
ax2.set_ylabel('Mean Episode Length', color='black')
plt.title('Mean Reward and Episode Length Progression over Iterations')

# Legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.grid(True)
plt.savefig("action_type_experiment_dual_y-axis.pdf")