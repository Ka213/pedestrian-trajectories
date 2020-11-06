from common_import import *

import datetime
from pathlib import Path
from multiprocessing import Pool
from my_learning.max_ent import *
from my_learning.learch import *
from my_learning.learch_loss_aug_esf import *
from my_learning.learch_esf import *
from my_learning.learch_avg_esf_path import *
from my_learning.only_push_down import *
from my_learning.random import *
from my_learning.nn_learning import *
from my_learning.irl import *
from my_learning.average import *
from my_utils.output_costmap import *
from my_utils.output_analysis import *


def parallel_task(learning, nb_train, nb_test, range_training_env,
                  range_test_env, workspace, nb_samples):
    nb_points = 28
    nb_rbfs = 4
    sigma = 0.15
    np.random.seed(0)
    print("# demonstrations: ", nb_samples)

    learning_time_0 = time.time()
    # Learn costmap
    if learning == 'learch':
        l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'maxEnt':
        l = MaxEnt(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'avg_esf_path':
        l = Learch_Avg_Esf_Path(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'loss_aug_esf':
        l = Learch_Loss_Aug_Esf(nb_points, nb_rbfs, sigma, workspace)
    elif learning == 'esf':
        l = Learch_Esf(nb_points, nb_rbfs, sigma, workspace)
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
            load_environment("environment_rbfs_28" + str(k))
        nb_points, nb_rbfs, sigma, _ = \
            load_environment_params("environment_rbfs_28" + str(k))
        original_costmaps.append(original_costmap)
        starts = s[:nb_samples]
        targets = t[:nb_samples]
        paths = p[:nb_samples]
        original_paths.append(p[:nb_train])
        original_starts.append(s[:nb_train])
        original_targets.append(t[:nb_train])
        # Learn costmap
        l.add_environment(centers, paths, starts, targets)

    learned_maps, _, w_t, step_count = l.solve()
    learning_time = time.time() - learning_time_0
    nb_steps = step_count
    optimal_paths = []
    for j, k in enumerate(l.instances):
        _, _, p = plan_paths(nb_train, learned_maps[j] -
                             np.min(learned_maps[j]), workspace,
                             starts=original_starts[j],
                             targets=original_targets[j])
        optimal_paths.append(p)

    # Calculate training loss
    training_loss_l = get_learch_loss(original_costmaps, optimal_paths,
                                      original_paths, nb_train)
    training_loss_m = get_maxEnt_loss(learned_maps, original_paths, nb_train)
    training_edt = get_edt_loss(nb_points, optimal_paths, original_paths, nb_train)
    training_costs = get_overall_loss(learned_maps, original_costmaps)
    training_nll = get_nll(optimal_paths, original_paths, nb_points, nb_train)

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
            "environment_rbfs_28" + str(k))
        p_original_maps.append(p_costmap)
        # Learned Costmap
        phi = get_phi(nb_points, p_centers, sigma, workspace)
        costmap = get_costmap(phi, w_t)
        p_learned_maps.append(costmap)
        p_starts.append(s[- nb_test:])
        p_targets.append(t[- nb_test:])
        p_paths.append(p[- nb_test:])
        _, _, p = plan_paths(nb_test, costmap - np.min(costmap), workspace,
                             starts=p_starts[-1], targets=p_targets[-1])
        predictions.append(p)
    prediction_time = time.time() - prediction_time_0

    # Calculate test loss
    test_loss_l = get_learch_loss(p_original_maps, predictions, p_paths,
                                  nb_test)
    test_loss_m = get_maxEnt_loss(p_learned_maps, p_paths, nb_test)
    test_nll = get_nll(predictions, p_paths, nb_points, nb_test)
    test_edt = get_edt_loss(nb_points, predictions, p_paths, nb_test)
    test_costs = get_overall_loss(p_learned_maps, p_original_maps)

    if learning == "learch":
        save_learch_params(directory + "/params", l)
    elif learning == "maxEnt":
        save_maxEnt_params(directory + "/params", l)
    elif learning == "occ":
        save_newAlg_params(directory + "/params", l)

    return learning_time, prediction_time, nb_steps, training_loss_l, \
           training_loss_m, training_edt, training_costs, training_nll, \
           test_loss_l, test_loss_m, test_nll, test_edt, test_costs


if __name__ == "__main__":
    show_result = 'SAVE'
    # set the learning method to evaluate
    # choose between learch, maxEnt, avg_esf_path, esf, loss_aug_esf
    # oneVector or random
    learning = 'random'
    nb_samples_l = 1
    nb_samples_u = 20
    step = 1
    range_training_env = np.arange(5)
    range_test_env = np.arange(5, 10)
    nb_train = 20
    nb_test = 20
    foldername = '{}_{}env_{}-{}samples_{}predictions' \
        .format(learning, len(range_training_env), nb_samples_l, nb_samples_u,
                nb_test)
    workspace = Workspace()
    directory = home + '/../results/prediction/' + foldername
    Path(directory).mkdir(parents=True, exist_ok=True)
    file = open(directory + "/metadata.txt", "w")
    file.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               + '\n')
    file.write("number of predictions: " + str(nb_test) + '\n')

    t_0 = time.time()
    pool = Pool()
    x = np.arange(nb_samples_l, nb_samples_u + 1, step)
    y = [(learning, nb_train, nb_test, range_training_env, range_test_env,
          workspace, i) for i in x]
    result = pool.starmap(parallel_task, y)
    learning_time = np.vstack(np.asarray(result)[:, 0])
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
    compare_learning([results], directory + '/output.pdf',
                     names=[learning], title=learning)

    file.write("duration: {}".format(time.time() - t_0) + 'sec \n')

    file.close()
