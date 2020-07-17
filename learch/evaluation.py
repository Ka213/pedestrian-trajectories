from common_import import *

import numpy as np
import datetime
import time
from my_utils.output import *
from my_utils.my_utils import *
from costmap.costmap import *
from learch import *
from pyrieef.geometry.workspace import *

show_result = 'SAVE'
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 0.1
step_size = 1
nb_samples_l = 1
nb_samples_u = 20
nb_runs = 1

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)

x = np.arange(nb_samples_l, nb_samples_u + 1)
# Data structure to measure the accuracy of the LEARCH algorithm
# Each row corresponds to one environment
# Columns i corresponds to i example trajectories used in the LEARCH algorithm
error_edt = np.zeros((nb_runs, len(x)))
error_cost = np.zeros((nb_runs, len(x)))
error_cost_along_path_gd = np.zeros((nb_runs, len(x)))

learned_maps = []
optimal_paths = []
weights = []
original_costmaps = []

# Open file to save weights of the environment
file = open(home + "/../environment.txt", "a")

file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
file.write(str(nb_runs) + '\n')

t = time.time()
for j in range(nb_runs):
    print("run: ", j)
    np.random.seed(j)

    # Create costmap with rbfs
    w = np.random.random(nb_rbfs ** 2)
    file.write(str(w) + '\n')
    centers = workspace.box.meshgrid_points(nb_rbfs)
    original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)
    original_costmaps.append(original_costmap)
    # Plan example trajectories
    starts, targets, paths = plan_paths(nb_samples_u,
                                        original_costmap, workspace)

    # Compute error
    for i in range(nb_samples_l, nb_samples_u + 1):
        # Learn costmap
        l = Learch2D(nb_points, centers, sigma, paths[:i],
                     starts[:i], targets[:i], workspace)
        learned_map, optimal_path, w = l.solve()
        learned_maps.append(learned_map[-1])
        optimal_paths.append(optimal_path[-1])
        weights.append(w[-1])

        try:
            # Calculate error between optimal and example paths
            for n, op in enumerate(optimal_path[-1]):
                error_edt[j, i - nb_samples_l] += \
                    get_edt(op, paths[n], nb_points) / len(paths[n])
                error_cost_along_path_gd[j, i - nb_samples_l] += ( \
                            np.abs(np.sum(np.exp(learned_map[-1])[np.asarray(op).astype(int)]) \
                                   - np.sum(np.exp(learned_map[-1])[np.asarray(paths[n])])) / len(paths[n]))

            error_edt[j, i - nb_samples_l] = error_edt[j, i - nb_samples_l] / i
            error_cost[j, i - nb_samples_l] = \
                np.sum(np.abs(learned_map[-1] - original_costmap)) / \
                (nb_points ** 2)
            error_cost_along_path_gd[j, i - nb_samples_l] = \
                error_cost_along_path_gd[j, i - nb_samples_l] / i

        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("run: ", j)
            break

# Plot Error from euclidean distance transform
directory = home + '/../figures/edt_{}runs_{}-{}samples_egd.png' \
    .format(nb_runs, nb_samples_l, nb_samples_u)
plot_error_avg(error_edt, x, nb_runs, directory)

# Plot Error from costs difference along the paths
directory = home + '/../figures/cost_along_paths_{}runs_{}-{}samples_egd.png' \
    .format(nb_runs, nb_samples_l, nb_samples_u)
plot_error_avg(error_cost_along_path_gd, x, nb_runs,
               directory)

# Plot Error from costs difference of the whole map
directory = home + '/../figures/cost_whole_map_{}runs_{}-{}samples_egd.png' \
    .format(nb_runs, nb_samples_l, nb_samples_u)
plot_error_avg(error_cost, x, nb_runs, directory)

# Show learned costmaps for different number of samples
show_multiple(learned_maps, original_costmap, workspace, show_result,
              step=step_size,
              directory=home + '/../figures/costmaps_{}runs_{}-{}samples_egd.png'
              .format(nb_runs, nb_samples_l, nb_samples_u))

duration = time.time() - t
file.write("duration: {}".format(duration) + '\n')

file.close()
