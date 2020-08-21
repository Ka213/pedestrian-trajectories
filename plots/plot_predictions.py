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

workspace = Workspace()
np.random.seed(1)
filename = "environment3"
# Load saved environment
w, original_costmap, s, t, p = load_environment(filename)
nb_points, nb_rbfs, sigma, _ = load_environment_params(filename)
centers = workspace.box.meshgrid_points(nb_rbfs)
# Traning set
starts = s[:nb_samples]
targets = t[:nb_samples]
paths = p[:nb_samples]
# Test set
p_starts = s[nb_samples + 1: nb_samples + 1 + nb_predictions]
p_targets = t[nb_samples + 1: nb_samples + 1 + nb_predictions]
p_paths = p[nb_samples + 1: nb_samples + 1 + nb_predictions]

# Learn costmap
if learning == 'learch':
    l = Learch2D(nb_points, centers, sigma, paths,
                 starts, targets, workspace)
    l.exponentiated_gd = True
    learned_map, optimal_path, w_t = l.solve()
elif learning == 'maxEnt':
    l = MaxEnt(nb_points, centers, sigma,
               paths, starts, targets, workspace)
    learned_map, w_t = l.solve()

# Predict paths
_, _, predictions = plan_paths(nb_predictions, learned_map[-1],
                               workspace, starts=p_starts,
                               targets=p_targets)
# Calculate test loss
if learning == 'learch':
    loss = get_learch_loss(original_costmap, predictions, p_paths, nb_samples,
                           l._l2_regularizer, l._proximal_regularizer, w_t[-1])
elif learning == 'maxEnt':
    loss = get_maxEnt_loss(learned_map[-1], p_paths, nb_samples, w_t[-1])

# Output predictions

show_iteration([learned_map[-1]], original_costmap, workspace, show_result,
               starts=p_starts, targets=p_targets, paths=p_paths,
               optimal_paths=[predictions], directory=home +
                                                      '/../figures/{}_{}predictions.png'.format(learning,
                                                                                                nb_predictions))

# show_multiple(learned_map, original_costmap, workspace, show_result,
#              directory= home +  '/../figures/{}_{}predictions.png'.format(
#                  learning, nb_predictions))
