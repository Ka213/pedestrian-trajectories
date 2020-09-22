import common_import

from my_learning.learch import *
from my_utils.environment import *
from my_utils.my_utils import *
from my_utils.output_costmap import *

show_result = 'SHOW'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 100
nb_env = 1

workspace = Workspace()

l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
original_costmaps = []
original_starts = []
original_targets = []
original_paths = []
for i in range(nb_env):
    np.random.seed(i)
    # Create random costmap
    w, original_costmap, starts, targets, paths, centers = \
        create_rand_env(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    original_costmaps.append(original_costmap)
    starts = starts[:nb_samples]
    targets = targets[:nb_samples]
    paths = paths[:nb_samples]
    original_paths.append(paths)
    original_starts.append(starts)
    original_targets.append(targets)
    # Learn costmap
    l.add_environment(centers, paths, starts, targets)
maps, optimal_paths, w_t, step = l.solve()

l2, l_proximal = l.get_regularization()
loss = get_learch_loss(original_costmaps, optimal_paths, original_paths,
                       nb_samples * nb_env, l2, l_proximal, w_t)
print("loss: ", loss)
loss = get_learch_loss(original_costmaps, optimal_paths, original_paths,
                       nb_samples * nb_env)
print("loss: ", np.average(loss))
training_edt = get_edt_loss(nb_points, optimal_paths, original_paths,
                            nb_samples * nb_env)
print("edt loss: ", np.average(training_edt))
error_cost = get_overall_loss(maps, original_costmaps)
print("cost loss: ", np.average(error_cost))

for j, i in enumerate(l.instances):
    for k in range(int(nb_samples / 5)):
        show_iteration(i.learned_maps, [original_costmaps[j]], workspace,
                       show_result, starts=original_starts[j][k * 5:(k + 1) * 5],
                       targets=original_targets[j][k * 5:(k + 1) * 5],
                       paths=original_paths[j][k * 5:(k + 1) * 5],
                       optimal_paths=np.asarray(i.optimal_paths)[:,
                                     k * 5:(k + 1) * 5])

# Output learned costmaps
show_multiple(maps, original_costmaps, workspace, show_result)
