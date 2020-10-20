from common_import import *

import datetime
from pathlib import Path
from multiprocessing import Pool
from my_learning.max_ent import *
from my_learning.learch import *
from my_learning.new_loss_aug_occ import *
from my_learning.new_occ import *
from my_learning.new_avg_occ_optpath import *
from my_learning.only_push_down import *
from my_learning.random import *
from my_learning.nn_learning import *
from my_learning.irl import *
from my_learning.average import *
from my_utils.output_costmap import *
from my_utils.output_analysis import *


def parallel_task(learning, nb_predictions, range_training_env, range_test_env,
                  workspace, i):
    print("# demonstrations: ", i)
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_training = 50

    learning_time_0 = time.time()
    # Learn costmap
    if learning == 'learch':
        l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'maxEnt':
        l = MaxEnt(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'new algorithm':
        l = New_Avg_Occ_OptPath(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'new algorithm2':
        l = New_Loss_Aug_Occ(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'new algorithm1':
        l = New_Occ(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'oneVector':
        l = Learning(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'average':
        l = Average(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'random':
        l = Random(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'onlyPushDown':
        l = OnlyPushDown(nb_points, nb_rbfs, sigma, workspace)
    if learning == 'nn_learch':
        l = NN_Learch(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'nn_maxEnt':
        l = NN_MaxEnt(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'nn_occ':
        l = NN_Occ(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'nn_loss_aug_occ':
        l = NN_Loss_Aug_Occ(nb_points, nb_rbfs, sigma, workspace)

    original_costmaps = []
    original_starts = []
    original_targets = []
    original_paths = []
    for k in range_training_env:
        # file.write("training environment: " + "environment" + str(k) + '\n')
        w, original_costmap, s, t, p, centers = \
            load_environment("environment_sample_centers" + str(k))
        nb_points, nb_rbfs, sigma, _ = \
            load_environment_params("environment_sample_centers" + str(k))
        original_costmaps.append(original_costmap)
        starts = s[:i]
        targets = t[:i]
        paths = p[:i]
        original_paths.append(p[:nb_training])
        original_starts.append(s[:nb_training])
        original_targets.append(t[:nb_training])
        # Learn costmap
        l.add_environment(centers, paths, starts, targets)

    learned_maps, _, w_t, step_count = l.solve()
    learning_time = time.time() - learning_time_0
    nb_steps = step_count
    optimal_paths = []
    for j, k in enumerate(l.instances):
        _, _, p = plan_paths(nb_training, learned_maps[j] -
                             np.amin(learned_maps[j]), workspace,
                             starts=original_starts[j],
                             targets=original_targets[j])
        optimal_paths.append(p)

    # Calculate training loss
    training_loss_l = get_learch_loss(original_costmaps, optimal_paths,
                                      original_paths, nb_training)
    training_loss_m = get_maxEnt_loss(learned_maps, original_paths, nb_training)
    training_edt = get_edt_loss(nb_points, optimal_paths, original_paths, nb_training)
    training_costs = get_overall_loss(learned_maps, original_costmaps)
    training_nll = get_nll(optimal_paths, original_paths, nb_points, nb_training)

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
        phi = get_phi(nb_points, p_centers, sigma, workspace)
        costmap = get_costmap(phi, w_t)
        p_learned_maps.append(costmap)
        p_starts.append(s[300:300 + nb_predictions])
        p_targets.append(t[300:300 + nb_predictions])
        p_paths.append(p[300:300 + nb_predictions])
        _, _, p = plan_paths(nb_predictions, costmap - np.amin(costmap), workspace,
                             starts=p_starts[-1], targets=p_targets[-1])
        predictions.append(p)
    prediction_time = time.time() - prediction_time_0

    # Calculate test loss
    test_loss_l = get_learch_loss(p_original_maps, predictions, p_paths,
                                  nb_predictions)
    test_loss_m = get_maxEnt_loss(p_learned_maps, p_paths, nb_predictions)
    test_nll = get_nll(predictions, p_paths, nb_points, nb_predictions)
    test_edt = get_edt_loss(nb_points, predictions, p_paths,
                            nb_predictions)
    test_costs = get_overall_loss(p_learned_maps, p_original_maps)

    if learning == "learch":
        save_learch_params(directory + "/params", l)
    elif learning == "maxEnt":
        save_maxEnt_params(directory + "/params", l)
    elif learning == "new algorithm":
        save_newAlg_params(directory + "/params", l)

    return learning_time, prediction_time, nb_steps, training_loss_l, \
           training_loss_m, training_edt, training_costs, training_nll, \
           test_loss_l, test_loss_m, test_nll, test_edt, test_costs


if __name__ == "__main__":
    show_result = 'SAVE'
    # set the learning method to evaluate
    # choose between learch, maxEnt, avg_occ_path, occ, loss_aug_occ
    # oneVector, random, nn_learch, nn_maxEnt, nn_occ and nn_loss_aug_occ
    learning = 'nn_learch'
    nb_samples_l = 1
    nb_samples_u = 20
    step = 1
    range_training_env = np.arange(5)
    nb_predictions = 50
    range_test_env = np.arange(5, 10)
    foldername = '{}_{}env_{}-{}samples_{}predictions' \
        .format(learning, len(range_training_env), nb_samples_l, nb_samples_u,
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
    y = [(learning, nb_predictions, range_training_env, range_test_env,
          workspace, i) for i in x]
    result = pool.starmap(parallel_task, y)
    learning_time = loss = np.vstack(np.asarray(result)[:, 0])
    prediction_time = np.vstack(np.asarray(result)[:, 1])
    nb_steps = np.vstack(np.asarray(result)[:, 2])
    training_loss_l = np.vstack(np.asarray(result)[:, 3])
    training_loss_m = np.vstack(np.asarray(result)[:, 4])
    training_edt = np.vstack(np.asarray(result)[:, 5])
    training_costs = np.vstack(np.asarray(result)[:, 6])
    training_nll = np.vstack(np.asarray(result)[:, 7])
    test_loss_l = np.vstack(np.asarray(result)[:, 8])
    test_loss_m = np.vstack(np.asarray(result)[:, 9])
    test_nll = np.vstack(np.asarray(result)[:, 10])
    test_edt = np.vstack(np.asarray(result)[:, 11])
    test_costs = np.vstack(np.asarray(result)[:, 12])

    pool.close()

    results = directory + '/results.npz'
    np.savez(results, x=x, learning_time=learning_time, prediction_time=
    prediction_time, nb_steps=nb_steps, training_loss_l=training_loss_l,
             training_loss_m=training_loss_m, training_edt=training_edt,
             training_costs=training_costs, training_nll=training_nll,
             test_loss_l=test_loss_l, test_loss_m=test_loss_m, test_nll=
             test_nll, test_edt=test_edt, test_costs=test_costs)

    # Plot results
    compare_learning([results], directory + '/output.png',
                     names=[learning], title=learning)

    file.write("duration: {}".format(time.time() - t_0) + 'sec \n')

    file.close()
