from common_import import *

import time
import datetime
from pathlib import Path
from my_utils.output_costmap import *
from my_utils.output_analysis import *
from my_utils.my_utils import *
from my_utils.environment import *
from my_learning.max_ent import *
from my_learning.learch import *

show_result = 'SAVE'
average_cost = False
# set the learning method to evaluate
# choose between learch and maxEnt
learning = 'learch'
nb_samples_l = 1
nb_samples_u = 2
nb_runs = 1
exponentiated_gd = True

workspace = Workspace()

x = np.arange(nb_samples_l, nb_samples_u + 1)
# Data structures to evaluate the learning algorithm
# Each row corresponds to one environment
# Columns i corresponds to i example trajectories used in the learning algorithm
error_edt = np.zeros((nb_runs, len(x)))
error_cost = np.zeros((nb_runs, len(x)))
loss = np.zeros((nb_runs, len(x)))
nb_steps = np.zeros((nb_runs, len(x)))
learning_time = np.zeros((nb_runs, len(x)))

learned_maps = []
optimal_paths = []
weights = []
original_costmaps = []

directory = home + '/../evaluation/{}_{}runs_{}-{}samples'.format(learning,
                                                                  nb_runs,
                                                                  nb_samples_l,
                                                                  nb_samples_u)
Path(directory).mkdir(parents=True, exist_ok=True)
file = open(directory + "/metadata.txt", "w")
file.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           + '\n')

t = time.time()
# For each environment
for j in range(nb_runs):
    print("run: ", j)
    file.write("run: " + str(j) + '\n')
    np.random.seed(j)
    w, original_costmap, starts, targets, paths = load_environment("environment"
                                                                   + str(j))
    nb_points, nb_rbfs, sigma, _ = load_environment_params(
        "environment" + str(j))
    pixel_map = workspace.pixel_map(nb_points)
    centers = workspace.box.meshgrid_points(nb_rbfs)
    file.write("environment: " + "environment" + str(j) + '\n')
    original_costmaps.append(original_costmap)

    # For each number of demonstrations
    for i in range(nb_samples_l, nb_samples_u + 1):
        print('{} samples'.format(i))
        time_0 = time.time()
        # Learn costmap
        if learning == 'learch':
            l = Learch2D(nb_points, centers, sigma, paths[:i],
                         starts[:i], targets[:i], workspace)
            l.exponentiated_gd = exponentiated_gd
            learned_map, optimal_path, w_t = l.solve()
        elif learning == 'maxEnt':
            l = MaxEnt(nb_points, centers, sigma,
                       paths[:i], starts[:i], targets[:i], workspace)
            learned_map, w_t = l.solve()
            _, _, optimal_path = plan_paths(i, learned_map[-1],
                                            workspace, starts=starts[:i],
                                            targets=targets[:i])
            optimal_path = [optimal_path]

        learning_time[j, i - nb_samples_l] = time.time() - time_0

        learned_maps.append(learned_map[-1])
        optimal_paths.append(optimal_path[-1])
        weights.append(w_t[-1])
        nb_steps[j, i - nb_samples_l] = len(w_t)

        # Calculate loss
        if learning == 'learch':
            loss[j, i - nb_samples_l] = get_learch_loss(original_costmap,
                                                        optimal_path[-1],
                                                        paths[:i], i,
                                                        l._l2_regularizer,
                                                        l._proximal_regularizer,
                                                        w_t[-1])
        elif learning == 'maxEnt':
            loss[j, i - nb_samples_l] = get_maxEnt_loss(learned_map[-1],
                                                        paths[:i], i, w_t[-1])

        error_edt[j, i - nb_samples_l] = get_edt_loss(nb_points,
                                                      optimal_paths[-1],
                                                      paths[:i], i)

        error_cost[j, i - nb_samples_l] = get_overall_loss(nb_points,
                                                           learned_map[-1],
                                                           original_costmap)

if learning == "learch":
    save_learch_params(directory + "/params", l)
elif learning == "maxEnt":
    save_maxEnt_params(directory + "/params", l)

# Plot results
plot_avg_over_runs(x, nb_runs, directory + "/edt.png", loss=error_edt,
                   nb_steps=nb_steps)
plot_avg_over_runs(x, nb_runs, directory + "/loss.png", loss=loss,
                   nb_steps=nb_steps)
plot_avg_over_runs(x, nb_runs, directory + "/costs.png", loss=error_cost,
                   nb_steps=nb_steps)
plot_avg_over_runs(x, nb_runs, directory + "/learning_time.png",
                   time=learning_time)
show_multiple(learned_maps, original_costmap, workspace, show_result,
              directory=directory + '/costmaps.png')

file.write("duration: {}".format(time.time() - t) + '\n')

file.close()
