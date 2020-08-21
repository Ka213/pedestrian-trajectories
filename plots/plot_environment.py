import common_import

from my_utils.environment import *
from my_utils.output_costmap import *
from my_utils.my_utils import *

show_result = 'SHOW'
with_trajectories = True
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 100

workspace = Workspace()
np.random.seed(1)
filename = "environment0"
# Load saved environment
w, costmap, starts, targets, paths = load_environment(filename)
nb_points, nb_rbfs, sigma, nb_samples = load_environment_params(filename)
print(nb_points, nb_rbfs, sigma, nb_samples)

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
