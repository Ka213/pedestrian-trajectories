from common_import import *

from pyrieef.geometry.workspace import *
import datetime
import time
from costmap.costmap import *
from my_utils.output import *
from my_utils.my_utils import *
from learch import *

show_result = 'SAVE'
# Choose the parameter: 'learning rate', 'step size scalar',
# 'l2 regularizer', 'proximal regurlarizer', 'exponentiated gradient descent'
# 'loss scalar' or 'loss stddev'
param = 'loss stddev'
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 10
nb_runs = 1
parameter_step = 1
param_upper_bound = 20
param_lower_bound = 1

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)

x = np.arange(param_lower_bound, param_upper_bound + 1, parameter_step)
# Data structure to measure the accuracy of the LEARCH algorithm
# Each row corresponds to one environment
# Columns i corresponds to parameter i used in the LEARCH algorithm
error_parameter = np.zeros((nb_runs, len(x)))
nb_steps = np.zeros((nb_runs, len(x)))

learned_maps = []
optimal_paths = []
weights = []

# Open file to save weights of the environment
file = open(home + "/../environment.txt", "a")

file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
file.write(str(nb_runs) + '\n')

t = time.time()
for j in range(nb_runs):
    try:
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

        for i in range(param_lower_bound, param_upper_bound + 1, parameter_step):
            # Learn costmap
            l = Learch2D(nb_points, centers, sigma, paths,
                         starts, targets, workspace)
            # Set hyperparameter to optimize
            if param == 'learning rate':
                l._learning_rate = i / 10
            elif param == 'step size scalar':
                l._stepsize_scalar = i
            elif param == 'l2 regurlarizer':
                l._l2_regularizer = i
            elif param == 'proximal regularizer':
                l._proximal_regularizer = i
            elif param == 'exponentiated gradient descent':
                l.exponentiated_gd = True
            elif param == 'loss scalar':
                l._loss_scalar = i
                l.initialize_mydata()
            elif param == 'loss stddev':
                l._loss_stddev = i
                l.initialize_mydata()

            learned_map, optimal_path, w_t = l.solve()
            nb_steps[j, int(i / parameter_step) - param_lower_bound] = (len(w_t))
            learned_maps.append(learned_map[-1])
            optimal_paths.append(optimal_path[-1])
            weights.append(w_t[-1])

            # Calculate error between optimal and example paths
            for n, op in enumerate(optimal_path[-1]):
                error_parameter[j, int(i / parameter_step) - param_lower_bound] += \
                    np.sum(original_costmap[np.asarray(paths[n]).T[:][0],
                                            np.asarray(paths[n]).T[:][1]]) \
                    - np.sum(original_costmap[np.asarray(op).T[:][0],
                                              np.asarray(op).T[:][1]])
            error_parameter[j, int(i / parameter_step) - param_lower_bound] = \
                error_parameter[j, int(i / parameter_step) - param_lower_bound] / \
                (nb_samples) + (l._l2_regularizer + l._proximal_regularizer) * \
                np.linalg.norm(w_t)
    except KeyboardInterrupt:
        print("here")
        break

# Plot Error from costs difference along the paths
directory = home + '/../figures/search_{}_{}l_{}u.png' \
    .format(param, param_lower_bound, param_upper_bound)
plot_error_avg(error_parameter, nb_steps, x, nb_runs, directory)

# Show learned costmaps for different number of samples
show_multiple(learned_maps, original_costmap, workspace, show_result,
              directory=home + '/../figures/costmaps_search_{}.png'.
              format(param))

duration = time.time() - t
file.write("duration: {}".format(duration) + '\n')

file.close()
