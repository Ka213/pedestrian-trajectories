from common_import import *

import time
import datetime
from pathlib import Path
from multiprocessing import Pool
from joblib import Parallel, delayed
from my_utils.output_costmap import *
from my_utils.output_analysis import *
from my_utils.my_utils import *
from my_utils.environment import *
from my_learning.max_ent import *
from my_learning.learch import *


def parallel_task(learning, nb_predictions, nb_env, range_test_env,
                  workspace, i):
    print("# demonstrations: ", i)
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1

    learning_time_0 = time.time()
    # Learn costmap
    if learning == 'learch':
        l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'maxEnt':
        l = MaxEnt(nb_points, nb_rbfs, sigma, workspace)
    original_costmaps = []
    original_starts = []
    original_targets = []
    original_paths = []
    for k in range(nb_env):
        # file.write("training environment: " + "environment" + str(k) + '\n')
        w, original_costmap, s, t, p, centers = \
            load_environment("environment_sample_centers" + str(k))
        nb_points, nb_rbfs, sigma, _ = load_environment_params("environment"
                                                               + str(k))
        original_costmaps.append(original_costmap)
        starts = s[:i]
        targets = t[:i]
        paths = p[:i]
        original_paths.append(paths)
        original_starts.append(starts)
        original_targets.append(targets)
        # Learn costmap
        l.add_environment(centers, paths, starts, targets)
    if learning == 'learch':
        learned_maps, optimal_paths, w_t, step_count = l.solve()
        l2, l_proximal = l.get_regularization()
    elif learning == 'maxEnt':
        learned_maps, optimal_paths, w_t, step_count = l.solve()
    learning_time = time.time() - learning_time_0
    nb_steps = step_count

    # Calculate training loss
    if learning == 'learch':
        training_loss_r = get_learch_loss(original_costmaps, optimal_paths, original_paths,
                                          i * nb_env, l2, l_proximal, w_t)
        training_loss = get_learch_loss(original_costmaps, optimal_paths, original_paths,
                                        i * nb_env)
    elif learning == 'maxEnt':
        training_loss_r = get_maxEnt_loss(learned_maps, original_paths, i * nb_env, w_t)
        training_loss = get_maxEnt_loss(learned_maps, original_paths, i * nb_env)

    training_edt = get_edt_loss(nb_points, optimal_paths, original_paths,
                                i * nb_env)
    error_cost = get_overall_loss(learned_maps, original_costmaps)
    training_nll = get_nll(optimal_paths, original_paths, nb_points, i * nb_env)

    # Predict paths
    prediction_time_0 = time.time()
    predictions = []
    p_original_maps = []
    p_learned_maps = []
    p_starts = []
    p_targets = []
    p_paths = []
    for k in range_test_env:
        # Test Environments
        # file.write("test environment: " + "environment" + str(k) + '\n')
        p_w, p_costmap, s, t, p, p_centers = load_environment(
            "environment_sample_centers" + str(k))
        p_original_maps.append(p_costmap)
        # Learned Costmap
        costmap = np.tensordot(w_t, get_phi(nb_points, p_centers, sigma,
                                            workspace), axes=1)
        p_learned_maps.append(costmap)
        p_starts.append(s[i:i + nb_predictions])
        p_targets.append(t[i:i + nb_predictions])
        p_paths.append(p[i:i + nb_predictions])
        _, _, p = plan_paths(nb_predictions, costmap - np.amin(costmap), workspace,
                             starts=p_starts[-1], targets=p_targets[-1])
        predictions.append(p)
    prediction_time = time.time() - prediction_time_0

    # Calculate test loss
    if learning == 'learch':
        test_loss_r = get_learch_loss(p_original_maps, predictions, p_paths,
                                      nb_predictions * nb_env, l2, l_proximal, w_t)
        test_loss = get_learch_loss(p_original_maps, predictions, p_paths,
                                    nb_predictions * nb_env)
    elif learning == 'maxEnt':
        test_loss_r = get_maxEnt_loss(p_learned_maps, p_paths,
                                      nb_predictions * nb_env, w_t)
        test_loss = get_maxEnt_loss(p_learned_maps, p_paths,
                                    nb_predictions * nb_env)
    test_nll = get_nll(predictions, p_paths, nb_points, nb_predictions * nb_env)
    test_edt = get_edt_loss(nb_points, predictions, p_paths,
                            nb_predictions * nb_env)

    if learning == "learch":
        save_learch_params(directory + "/params", l)
    elif learning == "maxEnt":
        save_maxEnt_params(directory + "/params", l)

    return learning_time, nb_steps, training_loss, training_edt, error_cost, \
           prediction_time, test_loss, test_nll, test_edt, training_nll, \
           training_loss_r, test_loss_r


if __name__ == "__main__":
    show_result = 'SAVE'
    # set the learning method to evaluate
    # choose between learch and maxEnt

    nb_samples_l = 5
    nb_samples_u = 5
    step = 1
    nb_environments = 1
    learning = 'learch'
    nb_predictions = 100
    range_test_env = np.arange(1)
    foldername = '{}_{}env_{}-{}samples_{}predictions'.format(learning,
                                                              nb_environments,
                                                              nb_samples_l,
                                                              nb_samples_u,
                                                              nb_predictions)
    workspace = Workspace()
    directory = home + '/../results/prediction/' + foldername
    Path(directory).mkdir(parents=True, exist_ok=True)
    file = open(directory + "/metadata.txt", "w")
    file.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               + '\n')
    file.write("number of predictions: " + str(nb_predictions) + '\n')

    t_0 = time.time()
    pool = Pool()
    x = np.arange(nb_samples_l, nb_samples_u + 1, step)
    y = [(learning, nb_predictions, nb_environments, range_test_env,
          workspace, i) for i in x]
    result = pool.starmap(parallel_task, y)
    learning_time = np.asarray(result)[:, 0]
    nb_steps = np.asarray(result)[:, 1]
    training_loss = np.asarray(result)[:, 2]
    training_edt = np.asarray(result)[:, 3]
    error_cost = np.asarray(result)[:, 4]
    prediction_time = np.asarray(result)[:, 5]
    test_loss = np.asarray(result)[:, 6]
    test_nll = np.asarray(result)[:, 7]
    test_edt = np.asarray(result)[:, 8]
    training_nll = np.asarray(result)[:, 9]
    training_loss_r = np.asarray(result)[:, 10]
    test_loss_r = np.asarray(result)[:, 11]

    pool.close()

    results = directory + '/results.npz'
    np.savez(results, x=x, test_nll=test_nll, training_nll=training_nll,
             test_loss=test_loss, training_loss=training_loss,
             test_edt=test_edt, training_edt=training_edt, costs=error_cost,
             nb_steps=nb_steps, learning_time=learning_time,
             prediction_time=prediction_time, training_loss_r=training_loss_r,
             test_loss_r=test_loss_r)

    # Plot results
    plot_avg_over_runs(x, 1, directory + "/test_nll.png", loss=test_nll)
    plot_avg_over_runs(x, 1, directory + "/training_nll.png", loss=training_nll)
    plot_avg_over_runs(x, 1, directory + "/test_loss.png", loss=test_loss)
    plot_avg_over_runs(x, 1, directory + "/test_loss_r.png", loss=test_loss_r)
    plot_avg_over_runs(x, 1, directory + "/training_loss.png",
                       loss=training_loss)
    plot_avg_over_runs(x, 1, directory + "/training_loss_r.png",
                       loss=training_loss_r)
    plot_avg_over_runs(x, 1, directory + "/training_edt.png", loss=training_edt)
    plot_avg_over_runs(x, 1, directory + "/test_edt.png", loss=test_edt)
    plot_avg_over_runs(x, 1, directory + "/costs.png", loss=error_cost)
    plot_avg_over_runs(x, 1, directory + "/nb_steps.png", nb_steps=nb_steps)
    plot_avg_over_runs(x, 1, directory + "/learning_time.png",
                       time=learning_time)
    plot_avg_over_runs(x, 1, directory + "/prediction_time.png",
                       prediction_time=prediction_time)

    file.write("duration: {}".format(time.time() - t_0) + 'sec \n')

    file.close()
