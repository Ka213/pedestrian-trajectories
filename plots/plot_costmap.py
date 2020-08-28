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

w, costmap, starts, targets, paths = \
    create_random_environment(nb_points, nb_rbfs, sigma, nb_samples,
                              workspace)

show_3D(costmap, workspace, show_result, starts=starts[:nb_samples],
        targets=targets[:nb_samples], paths=paths[:nb_samples])

# Output costmap
if with_trajectories:
    title = 'costmap from {} differently weighted radial basis ' \
            'functions with sample trajectories'.format(nb_rbfs)
    directory = home + '../results/figures/costmap_{}rbf_with_samples.png'. \
        format(nb_rbfs)
    show(costmap, workspace, show_result, directory=directory,
         starts=starts, targets=targets, paths=paths)
else:
    title = 'costmap from {} differently weighted ' \
            'radial basis functions'.format(nb_rbfs)
    directory = home + '../results/figures/costmap_{}rbf.png'.format(nb_rbfs)
    show(costmap, workspace, show_result, directory=directory, title=title)
