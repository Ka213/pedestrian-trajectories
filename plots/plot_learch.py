import common_import

from my_learning.learch import *
from my_utils.costmap import *
from my_utils.my_utils import *
from my_utils.output import *

show_result = 'SHOW'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 10
nb_steps = 10

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)

# Create environment with rbfs
w = np.random.random(nb_rbfs ** 2)
centers = workspace.box.meshgrid_points(nb_rbfs)
original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)

# Plan example trajectories
starts, targets, paths = plan_paths(nb_samples, original_costmap, workspace)

# Learn analysis
l = Learch2D(nb_points, centers, sigma, paths, starts, targets, workspace)

maps, optimal_paths, w = l.solve()  # n_step(nb_steps)

# Output learned costmaps
show(maps, original_costmap, workspace, show_result)
show_multiple(maps, original_costmap, workspace, show_result, starts=starts,
              targets=targets, paths=paths, optimal_paths=optimal_paths)
show_iteration(maps, original_costmap, workspace, show_result, starts=starts,
               targets=targets, paths=paths, optimal_paths=optimal_paths, weights=w)
