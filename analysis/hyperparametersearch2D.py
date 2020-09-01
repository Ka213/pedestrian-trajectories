from common_import import *

from pyrieef.geometry.workspace import *
import datetime
from my_utils.environment import *
from my_utils.output_costmap import *
from my_utils.my_utils import *
from my_learning.max_ent import *
from my_learning.learch import *

show_result = 'SAVE'
# Choose the parameter: 'step size', 'loss' or 'regularization'
param = 'step size'
# set the learning method to evaluate
# choose between learch and maxEnt
learning = 'learch'
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 10
nb_runs = 1
parameter_step = 1
param_lower_bound = 1
param_upper_bound = 2
exponentiated_gd = True

workspace = Workspace()

x1 = np.arange(param_lower_bound, param_upper_bound + 1, parameter_step)
x2 = np.arange(param_lower_bound, param_upper_bound + 1, parameter_step)
# Data structure to measure the accuracy of the LEARCH algorithm
# Each row corresponds to one environment
# Columns i corresponds to parameter i used in the LEARCH algorithm
loss = np.zeros((nb_runs, len(x1), len(x2)))
nb_steps = np.zeros((nb_runs, len(x1), len(x2)))
learning_time = np.zeros((nb_runs, len(x1), len(x2)))

learned_maps = []
optimal_paths = []
weights = []

# Open file to save weights of the environment
file = open(home + "/../environment.txt", "a")

file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
file.write(str(nb_runs) + '\n')

data = home + '/../results/hyperparametersearch2D/{}_search_{}_{}r_{}l_{}u.npz' \
    .format(learning, param, nb_runs, param_lower_bound, param_upper_bound)
t = time.time()
for j in range(nb_runs):
    print("run: ", j)
    np.random.seed(j)

    w, original_costmap, starts, targets, paths, centers = \
        create_rand_env(nb_points, nb_rbfs, sigma, nb_samples,
                        workspace)
    file.write(str(w) + '\n')

    for i in range(param_lower_bound, param_upper_bound + 1, parameter_step):
        for k in range(param_lower_bound, param_upper_bound + 1, parameter_step):
            print('{}, {} parameter'.format(i, k))
            try:
                if learning == 'learch':
                    l = Learch2D(nb_points, centers, sigma, paths,
                                 starts, targets, workspace)
                    l.exponentiated_gd = exponentiated_gd
                elif learning == 'maxEnt':
                    l = MaxEnt(nb_points, centers, sigma,
                               paths, starts, targets, workspace)
                # Set hyperparameter to optimize
                if param == 'step size':
                    l._learning_rate = i / 10
                    l._stepsize_scalar = k
                elif param == 'regularization':
                    l._l2_regularizer = 10 / (10 ** i)
                    l._proximal_regularizer = 10 / (10 ** k)
                elif param == 'loss':
                    l._loss_stddev = i
                    l._loss_scalar = k  # / 10
                    l.initialize_mydata()

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

                learning_time[j, int(i / parameter_step) - param_lower_bound,
                              int(k / parameter_step) - param_lower_bound] = \
                    time.time() - time_0
                nb_steps[j, int(i / parameter_step) - param_lower_bound,
                         int(k / parameter_step) - param_lower_bound] = \
                    (len(w_t))
                learned_maps.append(learned_map[-1])
                optimal_paths.append(optimal_path[-1])
                weights.append(w_t[-1])

                # Calculate loss
                x = int(i / parameter_step) - param_lower_bound
                y = int(k / parameter_step) - param_lower_bound
                if learning == 'learch':
                    loss[j, x, y] = get_learch_loss(original_costmap,
                                                    optimal_path[-1], paths,
                                                    nb_samples,
                                                    l._l2_regularizer,
                                                    l._proximal_regularizer,
                                                    w_t[-1])
                elif learning == 'maxEnt':
                    loss[j, x, y] = get_maxEnt_loss(learned_map[-1], paths,
                                                    nb_samples, w_t[-1])

            except KeyboardInterrupt:
                break
            except Exception:
                print("Exception happend")
                continue

if param == 'regularization':
    x1 = (10 / np.power(10, x1))
    x2 = (10 / np.power(10, x2))
    x1 = np.flipud(x1)
    x2 = np.flipud(x2)
    # loss = np.flipud(loss)
elif param == 'step size':
    x2 = x2 / 10
# elif param == 'loss':
#    x1 = x1 / 10

duration = time.time() - t
file.write("duration: {}".format(duration) + '\n')

np.savez(data, loss=loss, nb_steps=nb_steps, x1=x1, x2=x2,
         nb_runs=np.asarray(nb_runs))

file.close()
