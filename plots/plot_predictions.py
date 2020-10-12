import common_import

from my_utils.environment import *
from my_utils.output_costmap import *
from my_utils.my_utils import *
from my_learning.learch import *
from my_learning.max_ent import *

show_result = 'SAVE'
learning = 'learch'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 100
nb_predictions = 5
# TODO update to multiple environments
workspace = Workspace()
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
    w, original_costmap, s, t, p, centers = \
        load_environment("environment_sample_centers" + str(k))
    nb_points, nb_rbfs, sigma, _ = \
        load_environment_params("environment_sample_centers" + str(k))
    original_costmaps.append(original_costmap)
    starts = s[:nb_samples]
    targets = t[:nb_samples]
    paths = p[:nb_samples]
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

# Predict paths
predictions = []
p_original_maps = []
p_learned_maps = []
p_starts = []
p_targets = []
p_paths = []

# Test Environments
p_w, p_costmap, s, t, p, p_centers = load_environment(
    "environment_sample_centers0")
p_original_maps.append(p_costmap)
# Learned Costmap
phi = get_phi(nb_points, p_centers, sigma, workspace)
costmap = costmap = get_costmap(phi, w_t)
x = nb_samples
p_learned_maps.append(costmap)
p_starts = s[x:x + nb_predictions]
p_targets = t[x:x + nb_predictions]
p_paths = p[x:x + nb_predictions]
_, _, p = plan_paths(nb_predictions, costmap - np.amin(costmap), workspace,
                     starts=p_starts, targets=p_targets)
predictions.append(p)

# Calculate test loss
if learning == 'learch':
    test_loss = get_learch_loss(p_original_maps, predictions, [p_paths],
                                nb_predictions)
elif learning == 'maxEnt':
    test_loss = get_maxEnt_loss(p_learned_maps, [p_paths],
                                nb_predictions)
print("loss: ", test_loss)
test_nll = get_nll(predictions, [p_paths], nb_points, nb_predictions)
print("nll: ", test_nll)
test_edt = get_edt_loss(nb_points, predictions, [p_paths],
                        nb_predictions)
print("edt: ", test_edt)

# Output predictions
for i in range(20):
    show_iteration(p_learned_maps, p_original_maps, workspace, show_result,
                   starts=p_starts[i * 5:5 * (i + 1)], targets=p_targets[i * 5:5 * (i + 1)],
                   paths=p_paths[i * 5:5 * (i + 1)],
                   optimal_paths=[p[i * 5:5 * (i + 1)]], directory=
                   home + '/../results/figures/{}_{}predictions.png'.
                   format(learning, nb_predictions))

# show_multiple(learned_map, original_costmap, workspace, show_result,
#              directory= home +  '/../results/figures/{}_{}predictions.png'.
#              format(learning, nb_predictions))
