import pickle

from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate


with open("minigames/move_to_beacon/optuna/3/study.pkl", "rb") as f:
    study = pickle.load(f)

fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)
fig3 = plot_parallel_coordinate(study)

fig1.show()
fig2.show()
fig3.show()
