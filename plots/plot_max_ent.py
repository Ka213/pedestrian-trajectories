import common_import

from my_learning.max_ent import *
from my_utils.output_costmap import *
from my_utils.environment import *
from my_utils.my_utils import *

show_result = 'SHOW'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 1
nb_steps = 5

workspace = Workspace()
np.random.seed(1)
# Create random costmap
w, original_costmap, starts, targets, paths = \
    create_random_environment(nb_points, nb_rbfs, sigma, nb_samples, workspace)
centers = workspace.box.meshgrid_points(nb_rbfs)
# show(original_costmap, workspace, show_result)

# Learn costmap
a = MaxEnt(nb_points, centers, sigma, paths, starts, targets, workspace)
learned_map, w_t = a.solve()
# Calculate training loss
_, _, optimal_path = plan_paths(nb_samples, learned_map[-1],
                                workspace, starts=starts,
                                targets=targets)
loss = get_maxEnt_loss(learned_map[-1], paths, nb_samples,
                       w_t[-1])
print("loss: ", loss)
# Output learned costmaps
show_multiple(learned_map, original_costmap, workspace, show_result,
              directory=home + '/../figures/maxEnt.png')
show(learned_map[-1], workspace, show_result, starts=starts, targets=targets,
     paths=paths, directory=home + '/../figures/maxEnt_with_demonstrations.png')
