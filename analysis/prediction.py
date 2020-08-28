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
# set the learning method to evaluate
# choose between learch and maxEnt
learning = 'learch'
nb_samples_l = 1
nb_samples_u = 100
nb_environments = 1
nb_predictions = 100
nb_runs = 1
range_test_env = np.arange(1, 2)
foldername = '{}_test'.format(learning, nb_environments,
                              nb_samples_l, nb_samples_u, nb_predictions)

workspace = Workspace()

x = np.arange(nb_samples_l, nb_samples_u + 1)
# Data structure to evaluate the learning algorithm
# Each row corresponds to one environment
# Columns i corresponds to i example trajectories used in the learning algorithm
test_edt = np.zeros((nb_runs, len(x)))
training_edt = np.zeros((nb_runs, len(x)))
error_cost = np.zeros((nb_runs, len(x)))
test_loss = np.zeros((nb_runs, len(x)))
training_loss = np.zeros((nb_runs, len(x)))
nb_steps = np.zeros((nb_runs, len(x)))
learning_time = np.zeros((nb_runs, len(x)))
prediction_time = np.zeros((nb_runs, len(x)))
nll = np.zeros((nb_runs, len(x)))

learned_maps = []
optimal_paths = []
weights = []
original_costmaps = []

directory = home + '/../results/prediction/' + foldername
Path(directory).mkdir(parents=True, exist_ok=True)
file = open(directory + "/metadata.txt", "w")
file.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           + '\n')
file.write("number of predictions: " + str(nb_predictions) + '\n')

t_0 = time.time()

for j in range(nb_runs):
    print("run: ", j)
    file.write("run: " + str(j) + '\n')

    starts = [None] * (nb_samples_u)
    targets = [None] * (nb_samples_u)
    paths = [None] * (nb_samples_u)
    r = math.floor(nb_samples_u / nb_environments)
    # Create training set
    for k in range(nb_environments):
        file.write("training environment: " + "environment" + str(k) + '\n')
        w, original_costmap, s, t, p = load_environment("environment" + str(k))
        nb_points, nb_rbfs, sigma, _ = load_environment_params("environment"
                                                               + str(k))
        centers = workspace.box.meshgrid_points(nb_rbfs)
        original_costmaps.append(original_costmap)
        # Add paths to training set
        starts[:nb_samples_u - r * k] = s[:r]
        targets[:nb_samples_u - r * k] = t[:r]
        paths[:nb_samples_u - r * k] = p[:r]

    # Create test set
    for k in range_test_env:
        print("k", k)
        file.write("test environment: " + "environment" + str(k) + '\n')
        p_w, p_costmap, p_starts, p_targets, p_paths = load_environment(
            "environment" + str(k))
        a = int(nb_samples_u / nb_environments)
        b = math.floor(nb_predictions / len(range_test_env))
        p_starts = p_starts[a:a + nb_predictions - (k - range_test_env[0]) * b]
        p_targets = p_targets[a:a + nb_predictions - (k - range_test_env[0]) * b]
        p_paths = p_paths[a:a + nb_predictions - (k - range_test_env[0]) * b]

    # For each number of demonstrations
    for i in range(nb_samples_l, nb_samples_u + 1):
        print('{} samples'.format(i))
        learning_time_0 = time.time()
        # Learn costmap
        if learning == 'learch':
            l = Learch2D(nb_points, centers, sigma, paths[:i],
                         starts[:i], targets[:i], workspace)
            l.exponentiated_gd = True
            learned_map, optimal_path, w_t = l.solve()
        elif learning == 'maxEnt':
            l = MaxEnt(nb_points, centers, sigma,
                       paths[:i], starts[:i], targets[:i], workspace)
            learned_map, w_t = l.solve()
            _, _, optimal_path = plan_paths(i, learned_map[-1],
                                            workspace, starts=starts[:i],
                                            targets=targets[:i])
            optimal_path = [optimal_path]

        learning_time[j, i - nb_samples_l] = time.time() - learning_time_0
        learned_maps.append(learned_map[-1])
        optimal_paths.append(optimal_path[-1])
        weights.append(w_t[-1])
        nb_steps[j, i - nb_samples_l] = len(w_t)

        # Calculate training loss
        if learning == 'learch':
            training_loss[j, i - nb_samples_l] = get_learch_loss(
                original_costmap, optimal_path[-1], paths[:i], i,
                l._l2_regularizer, l._proximal_regularizer, w_t[-1])
        elif learning == 'maxEnt':
            training_loss[j, i - nb_samples_l] = get_maxEnt_loss(learned_map[-1],
                                                                 paths[:i], i,
                                                                 w_t[-1])

        training_edt[j, i - nb_samples_l] = get_edt_loss(nb_points,
                                                         optimal_paths[-1],
                                                         paths[:i], i)
        error_cost[j, i - nb_samples_l] = get_overall_loss(nb_points,
                                                           learned_map[-1],
                                                           original_costmap)

        # Predict paths
        prediction_time_0 = time.time()
        _, _, predictions = plan_paths(nb_predictions, learned_map[-1],
                                       workspace, starts=p_starts,
                                       targets=p_targets)
        prediction_time[j, i - nb_samples_l] = time.time() - prediction_time_0

        # Calculate test loss
        if learning == 'learch':
            test_loss[j, i - nb_samples_l] = get_learch_loss(
                original_costmap, predictions, p_paths, i, l._l2_regularizer,
                l._proximal_regularizer, w_t[-1])
        elif learning == 'maxEnt':
            test_loss[j, i - nb_samples_l] = get_maxEnt_loss(learned_map[-1],
                                                             p_paths, i, w_t[-1])
        nll[j, i - nb_samples_l] = get_nll(predictions, p_paths, nb_points)
        test_edt[j, i - nb_samples_l] = get_edt_loss(nb_points, predictions,
                                                     p_paths, i)

        # if i % 20 == 0:
        #    show_iteration([learned_map[-1]], original_costmap, workspace,
        #                   show_result, starts=p_starts, targets=p_targets,
        #                   paths=p_paths, optimal_paths=[predictions],
        #                   directory= directory +
        #                   '/costmap{}_{}_with_predictions.png'.format(j, i))

if learning == "learch":
    save_learch_params(directory + "/params", l)
elif learning == "maxEnt":
    save_maxEnt_params(directory + "/params", l)

# Plot results
plot_avg_over_runs(x, nb_runs, directory + "/nll.png", loss=nll)
plot_avg_over_runs(x, nb_runs, directory + "/test_loss.png", loss=test_loss)
plot_avg_over_runs(x, nb_runs, directory + "/training_loss.png",
                   loss=training_loss)
plot_avg_over_runs(x, nb_runs, directory + "/training_edt.png", loss=test_edt)
plot_avg_over_runs(x, nb_runs, directory + "/test_edt.png", loss=training_edt)
plot_avg_over_runs(x, nb_runs, directory + "/costs.png", loss=error_cost)
plot_avg_over_runs(x, nb_runs, directory + "/nb_steps.png", nb_steps=nb_steps)
plot_avg_over_runs(x, nb_runs, directory + "/learning_time.png",
                   time=learning_time)
plot_avg_over_runs(x, nb_runs, directory + "/prediction_time.png",
                   prediction_time=prediction_time)
show_multiple(learned_maps, original_costmaps, workspace, show_result,
              directory=directory + '/costmaps.png')

results = directory + '/results.npz'
np.savez(results, x=x, nll=nll, test_loss=test_loss,
         training_loss=training_loss, test_edt=test_edt,
         training_edt=training_edt, costs=error_cost, nb_steps=nb_steps,
         learning_time=learning_time, prediction_time=prediction_time)

file.write("duration: {}".format(time.time() - t_0) + 'sec \n')

file.close()
