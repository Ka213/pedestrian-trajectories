import common_import

from my_utils.costmap import *
from my_utils.output import *
from my_utils.my_utils import *

show_result = 'SHOW'
with_trajectories = True
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 5

workspace = Workspace()
np.random.seed(1)

# Create costmap with rbfs
w = np.random.random(nb_rbfs ** 2)
centers = workspace.box.meshgrid_points(nb_rbfs)
costmap = get_costmap(nb_points, centers, sigma, w, workspace)

# Plan example trajectories
starts, targets, paths = plan_paths(nb_samples, costmap, workspace)

# Output costmap
if with_trajectories:
    title = 'costmap from {} differently weighted radial basis ' \
            'functions with sample trajectories'.format(nb_rbfs)
    directory = home + '../figures/costmap_{}rbf_with_samples.png'. \
        format(nb_rbfs)
    show(costmap, workspace, show_result, directory=directory,
         starts=starts, targets=targets, paths=paths)
else:
    title = 'costmap from {} differently weighted ' \
            'radial basis functions'.format(nb_rbfs)
    directory = home + '../figures/costmap_{}rbf.png'.format(nb_rbfs)
    show(costmap, workspace, show_result, directory=directory, title=title)
