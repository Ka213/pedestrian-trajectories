from common_import import *

import datetime
from pathlib import Path
from my_learning.max_ent import *
from my_learning.learch import *
from my_learning.deep_learning import *
from my_utils.output_analysis import *
from my_learning.one_vector_sdf import *
from my_learning.random_sdf import *


def parallel_task(learning, nb_train, nb_test, nb_train_env, nb_test_env,
                  workspaces, costmaps, nb_samples, directory):
    ''' Calculate result for one specific number of demonstrations '''
    np.random.seed(0)
    print("# demonstrations: ", nb_samples)
    box = EnvBox()
    lims = workspaces[0].lims
    box.dim[0] = lims[0][1] - lims[0][0]
    box.dim[1] = lims[1][1] - lims[1][0]
    box.origin[0] = box.dim[0] / 2.
    box.origin[1] = box.dim[1] / 2.
    workspace = Workspace(box)

    # Learn costmap
    if learning == 'oneVector':
        l = One_Vector_Sdf(dataset, learning, workspace)
    elif learning == 'random':
        l = Random_Sdf(dataset, learning, workspace)
    else:
        l = Deep_Learning(costmaps, learning, nb_samples, workspaces)

    assert nb_train <= len(l.costmaps.train_inputs)
    assert nb_test <= len(l.costmaps.test_inputs)

    x = l.NUM_TRAIN

    # Get training samples
    costs = []
    demonstrations = []
    starts = []
    targets = []
    for j in range(nb_train_env):
        costs.append(l.workspaces[j].costmap)
        demonstrations.append(l.workspaces[j].demonstrations[:nb_train])
        starts.append(l.workspaces[j].starts[:nb_train])
        targets.append(l.workspaces[j].targets[:nb_train])

    # Get test samples
    p_costs = []
    p_demonstrations = []
    p_starts = []
    p_targets = []
    for j in range(nb_test_env):
        p_costs.append(l.workspaces[x + j].costmap)
        p_demonstrations.append(l.workspaces[x + j].demonstrations[:nb_test])
        p_starts.append(l.workspaces[x + j].starts[:nb_test])
        p_targets.append(l.workspaces[x + j].targets[:nb_test])

    nb_points = costs[0].shape[0]

    # Training
    learning_time_0 = time.time()
    learned_maps, _, step_count = l.solve()
    learning_time = time.time() - learning_time_0

    nb_steps = step_count

    ex_paths = []
    for j in range(nb_train_env):
        _, _, p = plan_paths(nb_samples, learned_maps[j], workspace,
                             starts=starts[j], targets=targets[j])
        ex_paths.append(p)

    # Calculate training loss
    training_loss_l = get_learch_loss(costs, ex_paths, demonstrations,
                                      nb_samples)
    training_loss_m = get_maxEnt_loss(learned_maps[:nb_train_env],
                                      demonstrations, nb_samples)
    training_edt = get_edt_loss(nb_points, ex_paths, demonstrations,
                                nb_samples)
    training_costs = get_overall_loss(learned_maps[:nb_train_env], costs)
    training_nll = get_nll(ex_paths, demonstrations, nb_points,
                           nb_samples)

    # Testing
    prediction_time_0 = time.time()
    p_paths = []
    for j in range(nb_test_env):
        _, _, p = plan_paths(nb_samples, learned_maps[x + j], workspace,
                             starts=p_starts[j], targets=p_targets[j])
        p_paths.append(p)
    prediction_time = time.time() - prediction_time_0

    # Calculate test loss
    test_loss_l = get_learch_loss(p_costs, p_paths, p_demonstrations,
                                  nb_samples)
    test_loss_m = get_maxEnt_loss(learned_maps[x:x + nb_test_env],
                                  p_demonstrations, nb_samples)
    test_nll = get_nll(p_paths, p_demonstrations, nb_points,
                       nb_samples)
    test_edt = get_edt_loss(nb_points, p_paths, p_demonstrations,
                            nb_samples)
    test_costs = get_overall_loss(learned_maps[x:x + nb_test_env], p_costs)

    save_results(directory + '/maps' + str(nb_samples), l.learned_maps,
                 l.decoded_data_train, l.decoded_data_test)

    if learning != 'random' and learning != 'oneVector':
        show_multiple(np.array(l.learned_maps)[:, l.NUM_TRAIN], [p_costs[0]],
                      workspace, 'SAVE', directory=directory + '/costmap'
                                                   + str(nb_samples))

        show_multiple(np.array(l.decoded_data_test)[:, 0].reshape((-1, nb_points,
                                                                   nb_points)),
                      [p_costs[0]], workspace, 'SAVE',
                      directory=directory + '/gradient' + str(nb_samples))

    return learning_time, prediction_time, nb_steps, training_loss_l, \
           training_loss_m, training_edt, training_costs, training_nll, \
           test_loss_l, test_loss_m, test_nll, test_edt, test_costs


if __name__ == "__main__":
    show_result = 'SAVE'
    # set the learning method to evaluate
    # choose learch, maxEnt, loss_aug_esf or esf
    learning = 'maxEnt'
    nb_samples = 25
    nb_train = 20
    nb_test = 20
    nb_train_env = 10
    nb_test_env = 10
    nn_steps = 100
    nb_env = 500
    step = 2

    dataset = 'sdf_data_' + str(nb_env) + '_' + str(nb_samples)
    foldername = '{}_{}samples_{}env_{}'.format(learning, nb_samples,
                                                nb_env, nn_steps)

    directory = home + '/../results/prediction_nn/' + foldername
    Path(directory).mkdir(parents=True, exist_ok=True)
    file = open(directory + "/metadata.txt", "w")
    file.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               + '\n')
    file.write("number of demonstrations: " + str(nb_samples) + '\n')

    t_0 = time.time()
    costmaps = get_dataset_id(data_id=dataset)
    workspaces = load_workspace_dataset(basename=dataset + '.hdf5')
    pool = Pool()
    x = np.arange(1, nb_samples + 1, step)
    y = [(learning, nb_train, nb_test, nb_train_env, nb_test_env, workspaces,
          costmaps, i, directory)
         for i in x]
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

    # Output results
    results = directory + '/results.npz'
    np.savez(results, x=x, learning_time=learning_time, prediction_time=
    prediction_time, nb_steps=nb_steps, training_loss_l=training_loss_l,
             training_loss_m=training_loss_m, training_edt=training_edt,
             training_costs=training_costs, training_nll=training_nll,
             test_loss_l=test_loss_l, test_loss_m=test_loss_m, test_nll=
             test_nll, test_edt=test_edt, test_costs=test_costs)

    compare_learning([results], directory + '/output.pdf',
                     names=[learning], title=learning)

    file.write("duration: {}".format(time.time() - t_0) + 'sec \n')

    file.close()
