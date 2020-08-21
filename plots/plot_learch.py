import common_import

from my_learning.learch import *
from my_utils.environment import *
from my_utils.my_utils import *
from my_utils.output_costmap import *

show_result = 'SHOW'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 10
nb_steps = 20

workspace = Workspace()
np.random.seed(1)
# Create random costmap
w, original_costmap, starts, targets, paths = \
    create_random_environment(nb_points, nb_rbfs, sigma, nb_samples, workspace)
centers = workspace.box.meshgrid_points(nb_rbfs)

# Learn costmap
l = Learch2D(nb_points, centers, sigma, paths, starts, targets, workspace)
maps, optimal_paths, w_t = l.solve()
# Calculate training loss
loss = get_learch_loss(original_costmap, optimal_paths[-1], paths, nb_samples,
                       l._l2_regularizer, l._proximal_regularizer, w_t[-1])
print("loss: ", loss)
# Output learned costmaps
show(maps[-1], workspace, show_result)
show(original_costmap, workspace, show_result, starts=starts,
     targets=targets, paths=paths, optimal_paths=optimal_paths[-1])
show_multiple(maps, original_costmap, workspace, show_result)
show_multiple(maps, original_costmap, workspace, show_result, starts=starts,
              targets=targets, paths=paths, optimal_paths=optimal_paths)
show_iteration(maps, original_costmap, workspace, show_result, starts=starts,
               targets=targets, paths=paths, optimal_paths=optimal_paths)
