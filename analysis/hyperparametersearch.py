from common_import import *

from pyrieef.geometry.workspace import *
import datetime
from my_utils.environment import *
from my_utils.output_costmap import *
from my_utils.output_analysis import *
from my_utils.my_utils import *
from my_learning.max_ent import *
from my_learning.learch import *

show_result = 'SAVE'
# Choose the parameter: 'learning rate', 'step size scalar',
# 'l2 regularizer', 'proximal regularizer',
# 'loss scalar', 'loss stddev' or 'N'
param = 'N'
# set the learning method to evaluate
# choose between learch and maxEnt
learning = 'maxEnt'
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 35
nb_runs = 2
parameter_step = 1
param_lower_bound = 1
param_upper_bound = 2
exponentiated_gd = True

workspace = Workspace()

x = np.arange(param_lower_bound, param_upper_bound + 1, parameter_step)
if param == 'l2 regularizer' or param == 'proximal regularizer':
    x = (1 / np.power(10, x))
elif param == 'learning rate' or param == 'step size scalar' \
        or param == 'loss scalar':
    x = x / 10
# Data structure to measure the accuracy of the LEARCH algorithm
# Each row corresponds to one environment
# Columns i corresponds to parameter i used in the LEARCH algorithm
loss = np.zeros((nb_runs, len(x)))
nb_steps = np.zeros((nb_runs, len(x)))
learning_time = np.zeros((nb_runs, len(x)))

learned_maps = []
optimal_paths = []
weights = []

# Open file to save weights of the environment
file = open(home + "/../environment.txt", "a")
file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
file.write(str(nb_runs) + '\n')

t = time.time()
for j in range(nb_runs):
    print("run: ", j)
    np.random.seed(j)

    w, original_costmap, starts, targets, paths = \
        create_random_environment(nb_points, nb_rbfs, sigma, nb_samples,
                                  workspace)
    file.write(str(w) + '\n')
    centers = workspace.box.meshgrid_points(nb_rbfs)

    for i in range(param_lower_bound, param_upper_bound + 1, parameter_step):
        print('{} parameter'.format(i))
        n = int(i / parameter_step) - param_lower_bound
        try:
            if learning == 'learch':
                l = Learch2D(nb_points, centers, sigma, paths,
                             starts, targets, workspace)
                l.exponentiated_gd = exponentiated_gd
            elif learning == 'maxEnt':
                l = MaxEnt(nb_points, centers, sigma,
                           paths, starts, targets, workspace)
            # Set hyperparameter to optimize
            if param == 'learning rate':
                l._learning_rate = i / 10
            elif param == 'step size scalar':
                l._stepsize_scalar = i / 10
            elif param == 'l2 regularizer':
                l._l2_regularizer = 1 / (10 ** i)
            elif param == 'proximal regularizer':
                l._proximal_regularizer = 1 / (10 ** i)
            elif param == 'loss scalar':
                l._loss_scalar = i / 10
                l.initialize_mydata()
            elif param == 'loss stddev':
                l._loss_stddev = i
                l.initialize_mydata()
            elif param == 'N':
                l._N = i

            time_0 = time.time()
            # Learn costmap
            if learning == 'learch':
                learned_map, optimal_path, w_t = l.solve()
            elif learning == 'maxEnt':
                learned_map, w_t = l.solve()
                _, _, optimal_path = plan_paths(nb_samples, learned_map[-1],
                                                workspace, starts=starts,
                                                targets=targets)
                optimal_path = [optimal_path]

            learning_time[j, int(i / parameter_step) - param_lower_bound] = \
                time.time() - time_0
            nb_steps[j, int(i / parameter_step) - param_lower_bound] = \
                (len(w_t))
            learned_maps.append(learned_map[-1])
            optimal_paths.append(optimal_path[-1])
            weights.append(w_t[-1])

            # Calculate loss
            if learning == 'learch':
                loss[j, n] = get_learch_loss(original_costmap, optimal_path[-1],
                                             paths, nb_samples, l._l2_regularizer,
                                             l._proximal_regularizer, w_t[-1])
            elif learning == 'maxEnt':
                loss[j, n] = get_maxEnt_loss(learned_map[-1], paths, nb_samples,
                                             w_t[-1])

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Exception happend with parameter value ", i)
            print(e)
            continue

if param == 'l2 regularizer' or param == 'proximal regularizer':
    x = np.flipud(x)
    loss = np.flipud(loss)

# Plot loss
directory = home + '/../figures/hyperparametersearch/search_{}_{}l_{}u.png' \
    .format(param, param_lower_bound, param_upper_bound)
plot_avg_over_runs(x, nb_runs, directory, loss=loss)

# Show learned costmaps for different number of samples
show_multiple(learned_maps, original_costmap, workspace, show_result,
              directory=home + '/../figures/hyperparametersearch/'
                               'costmaps_search_{}.png'.format(param))

duration = time.time() - t
file.write("duration: {}".format(duration) + '\n')

file.close()
