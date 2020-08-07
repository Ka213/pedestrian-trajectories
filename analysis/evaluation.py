from common_import import *

import datetime
from my_utils.output import *
from my_utils.my_utils import *
from my_utils.costmap import *
from my_learning.max_ent import *
from my_learning.learch import *

show_result = 'SAVE'
average_cost = False
# set the learning method to evaluate
# choose between learch and maxEnt
learning = 'learch'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples_l = 1
nb_samples_u = 10
nb_runs = 1
exponentiated_gd = True

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)

x = np.arange(nb_samples_l, nb_samples_u + 1)
# Data structure to measure the accuracy of the LEARCH algorithm
# Each row corresponds to one environment
# Columns i corresponds to i example trajectories used in the LEARCH algorithm
error_edt = np.zeros((nb_runs, len(x)))
error_cost = np.zeros((nb_runs, len(x)))
loss = np.zeros((nb_runs, len(x)))
nb_steps = np.zeros((nb_runs, len(x)))

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

    # Compute loss
    for i in range(nb_samples_l, nb_samples_u + 1):
        print('{} samples'.format(i))
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
            maps = [learned_map[-1]] * i
            optimal_path = [plan_paths_fix_start(starts[:i], targets[:i],
                                                 maps, workspace)]

        learned_maps.append(learned_map[-1])
        optimal_paths.append(optimal_path[-1])
        weights.append(w_t[-1])
        nb_steps[j, i - nb_samples_l] = len(w_t)

        # Calculate loss
        for n, op in enumerate(optimal_path[-1]):
            error_edt[j, i - nb_samples_l] += \
                get_edt(op, paths[n], nb_points) / len(paths[n])
            if learning == 'learch':
                loss[j, i - nb_samples_l] += \
                    np.sum(original_costmap[np.asarray(paths[n]).T[:][0],
                                            np.asarray(paths[n]).T[:][1]]) \
                    - np.sum(original_costmap[np.asarray(op).T[:][0],
                                              np.asarray(op).T[:][1]])
            elif learning == 'maxEnt':
                loss[j, i - nb_samples_l] += \
                    np.sum(learned_map[-1][np.asarray(paths[n]).T[:][0],
                                           np.asarray(paths[n]).T[:][1]])

        error_edt[j, i - nb_samples_l] = error_edt[j, i - nb_samples_l] / i
        error_cost[j, i - nb_samples_l] = \
            np.sum(np.abs(learned_map[-1] - original_costmap)) / \
            (nb_points ** 2)
        if learning == 'learch':
            loss[j, i - nb_samples_l] = (loss[j, i - nb_samples_l] / i) + \
                                        (l._l2_regularizer +
                                         l._proximal_regularizer) \
                                        * np.linalg.norm(w_t[-1])
        elif learning == 'maxEnt':
            loss[j, i - nb_samples_l] = (loss[j, i - nb_samples_l] / i) + \
                                        np.linalg.norm(w_t[-1])

# Plot Error from euclidean distance transform
directory = home + '/../figures/{}_edt_{}runs_{}-{}samples.png' \
    .format(learning, nb_runs, nb_samples_l, nb_samples_u)
plot_error_avg(error_edt, nb_steps, x, nb_runs, directory)

# Plot Error from costs difference along the paths
directory = home + '/../figures/{}_loss_{}runs_{}-{}samples_egd.png' \
    .format(learning, nb_runs, nb_samples_l, nb_samples_u)
plot_error_avg(loss, nb_steps, x, nb_runs,
               directory)

# Plot Error from costs difference of the whole map
directory = home + '/../figures/{}_cost_whole_map_{}runs_{}-{}samples_egd.png' \
    .format(learning, nb_runs, nb_samples_l, nb_samples_u)
plot_error_avg(error_cost, nb_steps, x, nb_runs, directory)

# Show learned costmaps for different number of samples
show_multiple(learned_maps, original_costmap, workspace, show_result,
              directory=home +
                        '/../figures/{}_costmaps_{}runs_{}-{}samples_egd.png'
              .format(learning, nb_runs, nb_samples_l, nb_samples_u))

duration = time.time() - t
file.write("duration: {}".format(duration) + '\n')

file.close()
