from common_import import *

import datetime
from pathlib import Path
from my_learning.max_ent import *
from my_learning.learch import *
from my_learning.deep_learning import *
from my_utils.output_analysis import *
from my_learning.one_vector_sdf import *
from my_learning.random_sdf import *
import matplotlib.pyplot as plt

show_result = 'SAVE'
# set the learning method to evaluate
# choose learch, maxEnt, loss_aug_esf or esf
learning = 'learch'
nb_samples = 20
nb_train = 20
nb_test = 20
nb_steps = 5
nn_steps = 100
nb_env = 500

dataset = 'sdf_data_' + str(nb_env) + '_' + str(nb_samples)
foldername = '{}_{}samples_{}steps_{}env_{}'.format(learning, nb_samples,
                                                    nb_steps, nb_env, nn_steps)

workspace = Workspace()
np.random.seed(0)
directory = home + '/../results/prediction_nn/' + foldername
Path(directory).mkdir(parents=True, exist_ok=True)
file = open(directory + "/metadata.txt", "w")
file.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           + '\n')
file.write("number of demonstrations: " + str(nb_samples) + '\n')

t_0 = time.time()

if learning == 'learch' or learning == 'maxEnt' or learning == 'loss_aug_esf':
    l = Deep_Learning(dataset, learning, workspace)
    l.NUM_TEST = min(nb_test, 5)
    l.initialize()
    l.BATCHES = nn_steps
elif learning == 'oneVector':
    l = One_Vector_Sdf(dataset, learning, workspace)
elif learning == 'random':
    l = Random_Sdf(dataset, learning, workspace)

box = EnvBox()
lims = l.workspaces[0].lims
box.dim[0] = lims[0][1] - lims[0][0]
box.dim[1] = lims[1][1] - lims[1][0]
box.origin[0] = box.dim[0] / 2.
box.origin[1] = box.dim[1] / 2.
workspace = Workspace(box)

assert nb_train <= len(l.costmaps.train_inputs)
assert nb_test <= len(l.costmaps.test_inputs)

learning_time = np.zeros(nb_steps)
prediction_time = np.zeros(nb_steps)
step_count = np.zeros(nb_steps)
training_loss_l = np.zeros((nb_steps, nb_train))
training_loss_m = np.zeros((nb_steps, nb_train))
training_edt = np.zeros((nb_steps, nb_train))
training_costs = np.zeros((nb_steps, nb_train))
training_nll = np.zeros((nb_steps, nb_train))
test_loss_l = np.zeros((nb_steps, nb_test))
test_loss_m = np.zeros((nb_steps, nb_test))
test_nll = np.zeros((nb_steps, nb_test))
test_edt = np.zeros((nb_steps, nb_test))
test_costs = np.zeros((nb_steps, nb_test))

x = l.NUM_TRAIN
costs = []
demonstrations = []
starts = []
targets = []
for j in range(nb_train):
    costs.append(l.workspaces[j].costmap)
    demonstrations.append(l.workspaces[j].demonstrations)
    starts.append(l.workspaces[j].starts)
    targets.append(l.workspaces[j].targets)

p_costs = []
p_demonstrations = []
p_starts = []
p_targets = []
for j in range(nb_test):
    p_costs.append(l.workspaces[x + j].costmap)
    p_demonstrations.append(l.workspaces[x + j].demonstrations)
    p_starts.append(l.workspaces[x + j].starts)
    p_targets.append(l.workspaces[x + j].targets)

nb_points = costs[0].shape[0]
# Training
learning_time_0 = time.time()
plt.ion()
for i in range(nb_steps):
    learned_maps, _, steps = l.n_steps(1, begin=i)
    learning_time[i] = time.time() - learning_time_0

    step_count[i] = steps

    optimal_paths = []
    for j in range(nb_train):
        _, _, op = plan_paths(nb_samples, learned_maps[j], workspace,
                              starts=starts[j], targets=targets[j])
        optimal_paths.append(op)

    # Calculate training loss
    training_loss_l[i] = get_learch_loss(costs, optimal_paths, demonstrations,
                                         nb_samples)
    training_loss_m[i] = get_maxEnt_loss(learned_maps[:nb_train],
                                         demonstrations, nb_samples)
    training_edt[i] = get_edt_loss(nb_points, optimal_paths, demonstrations,
                                   nb_samples)
    training_costs[i] = get_overall_loss(learned_maps[:nb_train], costs)
    training_nll[i] = get_nll(optimal_paths, demonstrations, nb_points,
                              nb_samples)

    # Testing
    prediction_time_0 = time.time()
    p_optimal_paths = []
    for j in range(nb_test):
        _, _, op = plan_paths(nb_samples, learned_maps[x + j], workspace,
                              starts=p_starts[j], targets=p_targets[j])
        p_optimal_paths.append(op)
    prediction_time[i] = time.time() - prediction_time_0

    # Calculate test loss
    test_loss_l[i] = get_learch_loss(p_costs, p_optimal_paths, p_demonstrations,
                                     nb_samples)
    test_loss_m[i] = get_maxEnt_loss(learned_maps[x:x + nb_test],
                                     p_demonstrations, nb_samples)
    test_nll[i] = get_nll(p_optimal_paths, p_demonstrations, nb_points,
                          nb_samples)
    test_edt[i] = get_edt_loss(nb_points, p_optimal_paths, p_demonstrations,
                               nb_samples)
    test_costs[i] = get_overall_loss(learned_maps[x:x + nb_test], p_costs)

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
                 x_label='Learch iterations')

if learning == 'learch' or learning == 'maxEnt' or learning == 'loss_aug_esf':
    show_multiple(np.array(l.learned_maps)[:, l.NUM_TRAIN], [p_costs[0]], workspace,
                  show_result, directory=directory + '/costmap')

    show_multiple(np.array(l.decoded_data_test)[:, 0].reshape((-1, nb_points,
                                                               nb_points)),
                  [p_costs[0]], workspace, show_result,
                  directory=directory + '/gradient')

file.write("duration: {}".format(time.time() - t_0) + 'sec \n')

file.close()
plt.ioff()
