import common_import

from my_utils.environment import *
from my_utils.output_costmap import *
from my_utils.my_utils import *
from nn.random_environment_sdf import *
from nn.dataset_sdf import *

show_result = 'SHOW'
with_trajectories = True
rbf = False
nb_points = 28
nb_rbfs = 4
sigma = 0.15
nb_samples = 5

workspace = Workspace()
np.random.seed(0)

if rbf:
    w, costmap, starts, targets, paths, centers = \
        create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)
else:
    workspaces = load_workspace_dataset(basename='sdf_data_100_10' + '.hdf5')
    costmap = workspaces[4].costmap
    starts = workspaces[4].starts[:nb_samples]
    targets = workspaces[4].targets[:nb_samples]
    paths = workspaces[4].demonstrations[:nb_samples]

# show_3D(costmap, workspace, show_result,
#        starts=starts, targets=targets, paths=paths
#        )

# Output costmap
if with_trajectories:
    title = 'costmap from {} differently weighted radial basis ' \
            'functions with sample trajectories'.format(nb_rbfs)
    directory = home + '/../results/figures/costmap_with_samples.pdf'. \
        format(nb_rbfs)
    show(costmap, workspace, show_result, directory=directory,
         starts=starts, targets=targets, paths=paths)
else:
    title = 'costmap from {} differently weighted ' \
            'radial basis functions'.format(nb_rbfs)
    directory = home + '../results/figures/costmap_{}rbf.pdf'.format(nb_rbfs)
    show(costmap, workspace, show_result, directory=directory, title=title)
