import common_import

from my_learning.max_ent import *
from my_utils.output import *
from my_utils.costmap import *
from my_utils.my_utils import *

show_result = 'SHOW'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 30
nb_steps = 5

workspace = Workspace()
np.random.seed(1)

# Create costmap with rbfs
w = np.random.random(nb_rbfs ** 2)
centers = workspace.box.meshgrid_points(nb_rbfs)
original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)
# show(original_costmap, workspace, show_result)

# Plan example trajectories
starts, targets, paths = plan_paths(nb_samples, original_costmap, workspace)

# Learn costmap
a = MaxEnt(nb_points, centers, sigma, paths, starts, targets, workspace)
maps, w = a.solve()
# Output learned costmaps
show_multiple(maps, original_costmap, workspace, show_result,
              directory=home + '/../figures/maxEnt.png')
