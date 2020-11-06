import common_import

from my_learning.learch_avg_esf_path import *
from my_learning.learch_esf import *
from my_learning.learch_loss_aug_esf import *
from my_learning.only_push_down import *
from my_learning.random import *
from my_learning.average import *
from my_learning.maxEnt_then_learch import *
from my_utils.environment import *
from my_utils.my_utils import *
from my_utils.output_costmap import *

show_result = 'SHOW'
# set the learning method to evaluate
# choose between learch, maxEnt, avg_esf_path, esf, loss_aug_esf
# oneVector and random
learning = 'loss_aug_esf'
nb_points = 28
nb_rbfs = 4
sigma = 0.1
nb_samples = 20
nb_env = 1
nb_training = 20

workspace = Workspace()

# Learn costmap
if learning == 'learch':
    l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'maxEnt':
    l = MaxEnt(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'avg_esf_path':
    l = Learch_Avg_Esf_Path(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'esf':
    l = Learch_Esf(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'loss_aug_esf':
    l = Learch_Loss_Aug_Esf(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'oneVector':
    l = Learning(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'average':
    l = Average(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'maxEntThenLearch':
    l = MaxEntThenLearch(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'random':
    l = Random(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'onlyPushDown':
    l = OnlyPushDown(nb_points, nb_rbfs, sigma, workspace)


original_costmaps = []
original_starts = []
original_targets = []
original_paths = []
for i in range(nb_env):
    np.random.seed(i)
    # Create random costmap
    w, original_costmap, s, t, p, centers = \
        create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    original_costmaps.append(original_costmap)
    starts = s[:nb_samples]
    targets = t[:nb_samples]
    paths = p[:nb_samples]
    original_paths.append(p[:nb_training])
    original_starts.append(s[:nb_training])
    original_targets.append(t[:nb_training])
    # Learn costmap
    l.add_environment(centers, paths, starts, targets)

learned_maps, optimal_paths, w_t, step_count = l.solve()

optimal_paths = []
for j, k in enumerate(l.instances):
    # save_results(home + '/../results/learning/nn_learch', k.learned_maps,
    #             k.optimal_paths, [None] * len(k.learned_maps), k.sample_starts,
    #             k.sample_targets, k.sample_trajectories)
    _, _, p = plan_paths(nb_training, learned_maps[j] -
                         np.amin(learned_maps[j]), workspace,
                         starts=original_starts[j],
                         targets=original_targets[j])
    optimal_paths.append(p)

# Calculate training loss
training_loss_l = get_learch_loss(original_costmaps, optimal_paths,
                                  original_paths, nb_training)
print("learch loss: ", np.average(training_loss_l))
training_loss_m = get_maxEnt_loss(learned_maps, original_paths, nb_training)
print("maxEnt loss: ", np.average(training_loss_m))
training_edt = get_edt_loss(nb_points, optimal_paths, original_paths, nb_training)
print("edt loss: ", np.average(training_edt))
training_costs = get_overall_loss(learned_maps, original_costmaps)
print("cost loss: ", np.average(training_costs))
training_nll = get_nll(optimal_paths, original_paths, nb_points, nb_training)
print("nll loss: ", np.average(training_nll))

for j, i in enumerate(l.instances):
    show_predictions(learned_maps[j], original_costmaps[j], workspace, show_result,
                     starts=original_starts[j], targets=original_targets[j],
                     paths=original_paths[j],
                     optimal_paths=optimal_paths[j])
    # for k in range(int(nb_samples / 5)):
    #    show_iteration(i.learned_maps, [original_costmaps[j]], workspace,
    #                   show_result, starts=original_starts[j][k * 5:(k + 1) * 5],
    #                   targets=original_targets[j][k * 5:(k + 1) * 5],
    #                   paths=original_paths[j][k * 5:(k + 1) * 5],
    #                   optimal_paths=np.asarray(i.optimal_paths)[:,
    #                                 k * 5:(k + 1) * 5])

# Output learned costmaps
show_multiple(learned_maps, original_costmaps, workspace, show_result)
