import common_import

from my_learning.max_ent import *
from my_learning.learch import *
from my_learning.new_algorithm import *
from my_learning.new_algorithm1 import *
from my_learning.irl import *
from my_learning.only_push_down import *
from my_learning.random import *
from my_utils.environment import *
from my_utils.my_utils import *
from my_utils.output_costmap import *

show_result = 'SHOW'
# set the learning method to evaluate
# choose between learch, maxEnt, new algorithm, new alorithm1, uniform, average or maxEntThenLearch
learning = 'new algorithm'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 20
nb_env = 1

workspace = Workspace()

# Learn costmap
if learning == 'learch':
    l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'maxEnt':
    l = MaxEnt(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'new algorithm':
    l = NewAlgorithm(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'new algorithm1':
    l = NewAlgorithm_1(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'oneVector':
    l = Learning(nb_points, nb_rbfs, sigma, workspace)
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
    w, original_costmap, starts, targets, paths, centers = \
        load_environment("environment_sample_centers" + str(i))  # \
    #    create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    original_costmaps.append(original_costmap)
    starts = starts[:nb_samples]
    targets = targets[:nb_samples]
    paths = paths[:nb_samples]
    original_paths.append(paths)
    original_starts.append(starts)
    original_targets.append(targets)
    # Learn costmap
    l.add_environment(centers, paths, starts, targets)

learned_maps, optimal_paths, w_t, step_count = l.solve()

# Calculate training loss
training_loss_l = get_learch_loss(original_costmaps, optimal_paths,
                                  original_paths, nb_samples)
print("learch loss: ", np.average(training_loss_l))
training_loss_m = get_maxEnt_loss(learned_maps, original_paths, nb_samples)
print("maxEnt loss: ", np.average(training_loss_m))
training_edt = get_edt_loss(nb_points, optimal_paths, original_paths, nb_samples)
print("edt loss: ", np.average(training_edt))
training_costs = get_overall_loss(learned_maps, original_costmaps)
print("cost loss: ", np.average(training_costs))
training_nll = get_nll(optimal_paths, original_paths, nb_points, nb_samples)
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
