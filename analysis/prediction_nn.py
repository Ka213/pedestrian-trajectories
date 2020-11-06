from common_import import *

import datetime
from pathlib import Path
from my_learning.max_ent import *
from my_learning.learch import *
from my_learning.nn_linear_learch import *
from my_utils.output_analysis import *

import tensorflow as tf

# print("cpu device name",tf.test.cpu_device_name())
# print("gpu device name",tf.test.gpu_device_name())
# print("is gpu available:", tf.test.is_gpu_available())
# print("build with gpu support:", tf.test.is_built_with_gpu_support())
# print("list devices: ", tf.config.experimental_list_devices())
# print("logiacal devices: ", tf.config.list_logical_devices())
# if tf.test.gpu_device_name():
#    print('Default GPU Device: {}'
#          .format(tf.test.gpu_device_name()))
# else:
#   print("Please install GPU version of TF")
# tf.debugging.set_log_device_placement(True)


show_result = '-'
# set the learning method to evaluate
# choose learch, maxEnt, occ and loss_aug_occ
learning = 'learch'
nb_samples = 20
nb_points = 28
nb_rbfs = 4
sigma = 0.15
nb_train_env = 100
nb_test_env = 100
nb_steps = 3
nn_steps = 100

dataset = 'learch_data_' + str(nb_samples)  # TODO remove comment
foldername = '{}_{}samples_{}steps'.format(learning, nb_samples, nb_steps)

workspace = Workspace()
np.random.seed(0)
directory = home + '/../results/prediction_nn/' + foldername
Path(directory).mkdir(parents=True, exist_ok=True)
file = open(directory + "/metadata.txt", "w")
file.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           + '\n')
file.write("number of demonstrations: " + str(nb_samples) + '\n')

t_0 = time.time()

if learning == 'learch':
    l = NN_Learch(nb_points, nb_rbfs, sigma, dataset, workspace)
    l.NUM_TEST = nb_train_env
    l.initialize()

l.BATCHES = nn_steps

assert nb_train_env <= len(l.costmaps.train_inputs)
assert nb_test_env <= len(l.costmaps.test_inputs)

learning_time = np.zeros(nb_steps)
prediction_time = np.zeros(nb_steps)
step_count = np.zeros(nb_steps)
training_loss_l = np.zeros((nb_steps, nb_train_env))
training_loss_m = np.zeros((nb_steps, nb_train_env))
training_edt = np.zeros((nb_steps, nb_train_env))
training_costs = np.zeros((nb_steps, nb_train_env))
training_nll = np.zeros((nb_steps, nb_train_env))
test_loss_l = np.zeros((nb_steps, nb_test_env))
test_loss_m = np.zeros((nb_steps, nb_test_env))
test_nll = np.zeros((nb_steps, nb_test_env))
test_edt = np.zeros((nb_steps, nb_test_env))
test_costs = np.zeros((nb_steps, nb_test_env))

x = l.NUM_TRAIN
costs = []
demonstrations = []
starts = []
targets = []
for j in range(nb_train_env):
    costs.append(l.workspaces[j].costmap)
    demonstrations.append(l.workspaces[j].demonstrations)
    starts.append(l.workspaces[j].starts)
    targets.append(l.workspaces[j].targets)

p_costs = []
p_demonstrations = []
p_starts = []
p_targets = []
for j in range(nb_test_env):
    p_costs.append(l.workspaces[x + j].costmap)
    p_demonstrations.append(l.workspaces[x + j].demonstrations)
    p_starts.append(l.workspaces[x + j].starts)
    p_targets.append(l.workspaces[x + j].targets)

# Training
learning_time_0 = time.time()
for i in range(nb_steps):
    learned_maps, _, steps = l.n_steps(1, begin=i)
    learning_time[i] = time.time() - learning_time_0

    step_count[i] = steps

    optimal_paths = []
    for j in range(nb_train_env):
        map = learned_maps[j] - np.min(learned_maps[j])
        _, _, op = plan_paths(nb_samples, map, workspace, starts=starts[j],
                              targets=targets[j])
        optimal_paths.append(op)

    # Calculate training loss
    training_loss_l[i] = get_learch_loss(costs, optimal_paths, demonstrations,
                                         nb_samples)
    training_loss_m[i] = get_maxEnt_loss(learned_maps[:nb_train_env],
                                         demonstrations, nb_samples)
    training_edt[i] = get_edt_loss(nb_points, optimal_paths, demonstrations,
                                   nb_samples)
    training_costs[i] = get_overall_loss(learned_maps[:nb_train_env], costs)
    training_nll[i] = get_nll(optimal_paths, demonstrations, nb_points,
                              nb_samples)

    # Testing
    prediction_time_0 = time.time()
    p_optimal_paths = []
    for j in range(nb_test_env):
        map = learned_maps[x + j] - learned_maps[x + j].min()
        _, _, op = plan_paths(nb_samples, map, workspace, starts=p_starts[j],
                              targets=p_targets[j])
        p_optimal_paths.append(op)
    prediction_time[i] = time.time() - prediction_time_0

    # Calculate test loss
    test_loss_l[i] = get_learch_loss(p_costs, p_optimal_paths, p_demonstrations,
                                     nb_samples)
    test_loss_m[i] = get_maxEnt_loss(learned_maps[x:x + nb_test_env],
                                     p_demonstrations, nb_samples)
    test_nll[i] = get_nll(p_optimal_paths, p_demonstrations, nb_points,
                          nb_samples)
    test_edt[i] = get_edt_loss(nb_points, p_optimal_paths, p_demonstrations,
                               nb_samples)
    test_costs[i] = get_overall_loss(learned_maps[x:x + nb_test_env], p_costs)

    show_multiple(learned_maps[x:x + nb_test_env], p_costs, workspace,
                  show_result, directory=directory + '/costmaps_step' + str(i))

    show_multiple(np.array(l.decoded_data_test)[-1, :nb_test_env]
                  .reshape((-1, nb_points, nb_points)), p_costs, workspace,
                  show_result, directory=directory + '/gradient_step' + str(i))

# Save analysis values in file
results = directory + '/results.npz'
x = np.arange(1, nb_steps + 1)
np.savez(results, x=x, learning_time=learning_time, prediction_time=
prediction_time, nb_steps=step_count, training_loss_l=training_loss_l,
         training_loss_m=training_loss_m, training_edt=training_edt,
         training_costs=training_costs, training_nll=training_nll,
         test_loss_l=test_loss_l, test_loss_m=test_loss_m, test_nll=
         test_nll, test_edt=test_edt, test_costs=test_costs)

save_results(directory + '/maps', l.learned_maps, l.decoded_data_train,
             l.decoded_data_test)

print(test_loss_l)
# Plot results
compare_learning([results], directory + '/output.pdf', names=[learning],
                 title=learning)

show_multiple(np.array(l.learned_maps)[:, l.NUM_TRAIN], [p_costs[0]], workspace,
              show_result, directory=directory + '/costmap_over_iterations')

file.write("duration: {}".format(time.time() - t_0) + 'sec \n')

file.close()
