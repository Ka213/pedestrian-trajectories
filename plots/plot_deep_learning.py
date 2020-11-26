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
learning = 'maxEnt'
nb_samples = 1
nb_train = 20
nb_test = 20
nb_steps = 1
nn_steps = 1
nb_env = 250

dataset = 'sdf_data_' + str(nb_env) + '_25'  # + str(nb_samples)
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
costmaps = get_dataset_id(data_id=dataset)
workspaces = load_workspace_dataset(basename=dataset + '.hdf5')

box = EnvBox()
lims = workspaces[0].lims
box.dim[0] = lims[0][1] - lims[0][0]
box.dim[1] = lims[1][1] - lims[1][0]
box.origin[0] = box.dim[0] / 2.
box.origin[1] = box.dim[1] / 2.
workspace = Workspace(box)

if learning == 'oneVector':
    l = One_Vector_Sdf(dataset, learning, workspace)
elif learning == 'random':
    l = Random_Sdf(dataset, learning, workspace)
else:
    l = Deep_Learning(costmaps, learning, nb_samples, workspaces)

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
# Get training samples
costs = []
demonstrations = []
starts = []
targets = []
for j in range(nb_train):
    costs.append(l.workspaces[j].costmap)
    demonstrations.append(l.workspaces[j].demonstrations)
    starts.append(l.workspaces[j].starts)
    targets.append(l.workspaces[j].targets)
# Get test samples
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
# Compute results in each iteration
for i in range(nb_steps):
    try:
        learned_maps, _, steps = l.n_steps(1, begin=i)
        learning_time[i] = time.time() - learning_time_0

        step_count[i] = steps

        ex_paths = []
        for j in range(nb_train):
            _, _, p = plan_paths(nb_samples, learned_maps[j], workspace,
                                 starts=starts[j], targets=targets[j])
            ex_paths.append(p)

        # Calculate training loss
        training_loss_l[i] = get_learch_loss(costs, ex_paths,
                                             demonstrations, nb_samples)
        training_loss_m[i] = get_maxEnt_loss(learned_maps[:nb_train],
                                             demonstrations, nb_samples)
        training_edt[i] = get_edt_loss(nb_points, ex_paths, demonstrations,
                                       nb_samples)
        training_costs[i] = get_overall_loss(learned_maps[:nb_train], costs)
        training_nll[i] = get_nll(ex_paths, demonstrations, nb_points,
                                  nb_samples)

        # Testing
        prediction_time_0 = time.time()
        p_paths = []
        for j in range(nb_test):
            _, _, p = plan_paths(nb_samples, learned_maps[x + j], workspace,
                                 starts=p_starts[j], targets=p_targets[j])
            p_paths.append(p)
        prediction_time[i] = time.time() - prediction_time_0

        # Calculate test loss
        test_loss_l[i] = get_learch_loss(p_costs, p_paths, p_demonstrations,
                                         nb_samples)
        test_loss_m[i] = get_maxEnt_loss(learned_maps[x:x + nb_test],
                                         p_demonstrations, nb_samples)
        test_nll[i] = get_nll(p_paths, p_demonstrations, nb_points, nb_samples)
        test_edt[i] = get_edt_loss(nb_points, p_paths, p_demonstrations,
                                   nb_samples)
        test_costs[i] = get_overall_loss(learned_maps[x:x + nb_test], p_costs)
    except Exception as e:
        print("Exception happened in step", i)
        print(e)
        break
    except KeyboardInterrupt:
        print("Keyboard interrupted in step", i)
        break

# Output results
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

compare_learning([results], directory + '/output.pdf', names=[learning],
                 x_label='Learch iterations')

if learning != 'random' and learning != 'oneVector':
    show_multiple(np.array(l.learned_maps)[:, l.NUM_TRAIN], [p_costs[0]],
                  workspace, show_result, directory=directory + '/costmap')

    show_multiple(np.array(l.decoded_data_test)[:, 0].reshape((-1, nb_points,
                                                               nb_points)),
                  [p_costs[0]], workspace, show_result,
                  directory=directory + '/gradient')

file.write("duration: {}".format(time.time() - t_0) + 'sec \n')

file.close()
plt.ioff()
