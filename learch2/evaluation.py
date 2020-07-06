#!/usr/bin/env python

# Copyright (c) 2018, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                        Jim Mainprice on Sunday June 13 2018
from common_import import *

from pyrieef.geometry.workspace import *
import datetime
from learch2.learch import *
from learch2.output import *


def get_edt(optimal_trajectory, sample_trajectory):
    """ Return the euclidean distance transform
        of the optimal trajectory to the example trajectory
    """
    occpancy_map = np.zeros((nb_points, nb_points))
    x_1 = np.asarray(sample_trajectory)[:, 0]
    x_2 = np.asarray(sample_trajectory)[:, 1]
    occpancy_map[x_1, x_2] = 1
    distance = edt(occpancy_map)
    return np.sum(distance[np.asarray(optimal_trajectory[0])])


show_result = True
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 3
nb_runs = 2

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)

# Data structure to measure the accuracy of the LEARCH algorithm
# Each row corresponds to one environment
# Columns i corresponds to i example trajectories used in the LEARCH algorithm
error_edt = np.zeros((nb_runs, nb_samples))
error_cost = np.zeros((nb_runs, nb_samples))
error_cost_along_path = np.zeros((nb_runs, nb_samples))

# Path to current directory
home = os.path.abspath(os.path.dirname(__file__))

# Open file to save weights of the environment
file = open(home + "/../environment.txt", "a")
file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
file.write(str(nb_runs) + '\n')

for j in range(nb_runs):
    print("run: ", j)
    np.random.seed(j)

    # Create costmap with rbfs
    w = np.random.random(nb_rbfs ** 2)
    file.write(str(w) + '\n')
    centers = workspace.box.meshgrid_points(nb_rbfs)
    original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)

    # Plan example trajectories
    starts, targets, paths = plan_paths(nb_samples,
                                        original_costmap, workspace)

    # Compute error
    for i in range(nb_samples):
        # Learn costmap
        l = Learch2D(nb_points, centers, sigma, paths[:i + 1],
                     starts[:i + 1], targets[:i + 1], workspace)
        learned_map, optimal_paths, w = l.solve()

        # Calculate error between optimal and example paths
        for n, op in enumerate(optimal_paths[-1]):
            error_edt[j, i] += get_edt(op, paths[n]) / len(op)
            error_cost[j, i] += np.sum(learned_map[-1][np.asarray(op).astype(int)]) \
                                - np.sum(original_costmap[np.asarray(paths[n])])

        error_edt[j, i] = error_edt[j, i] / (i + 1)
        error_cost_along_path[j, i] = np.sum(np.abs(learned_map - original_costmap)) / \
                           (nb_points * nb_points)

# Plot Error from euclidean distance transform
directory = home + '/../figures/edt_diff_nb_samples_avg_seeds.png'
plot_error_avg(error_edt, nb_samples, nb_runs, directory)
directory = home + '/../figures/edt_diff_nb_samples.png'
plot_error_fix_env(error_edt[0, :], nb_samples, directory)
directory = home + '/../figures/edt_diff_seeds.png'
plot_error_fix_nbsamples(error_edt[:, 1], 2, nb_runs, directory)

# Plot Error from costs
directory = home + '/../figures/cost_diff_nb_samples_avg_seeds.png'
plot_error_avg(error_cost, nb_samples, nb_runs, directory)
directory = home + '/../figures/cost_diff_nb_samples.png'
plot_error_fix_env(error_cost[0, :], nb_samples, directory)
directory = home + '/../figures/cost_diff_seeds.png'
plot_error_fix_nbsamples(error_cost[:, 1], 2, nb_runs, directory)

# Plot Error from costs difference along the paths
directory = home + '/../figures/cost_along_paths_diff_nb_samples_avg_seeds.png'
plot_error_avg(error_cost_along_path, nb_samples, nb_runs, directory)
directory = home + '/../figures/cost_along_paths_diff_nb_samples.png'
plot_error_fix_env(error_cost_along_path[0, :], nb_samples, directory)
directory = home + '/../figures/cost_along_paths_diff_seeds.png'
plot_error_fix_nbsamples(error_cost_along_path[:, 1], 2, nb_runs, directory)

file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
file.close()
