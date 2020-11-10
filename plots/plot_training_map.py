import common_import

from nn.dataset_sdf import *
from my_utils.output_costmap import *
from my_utils.my_utils import *
from nn.nn_utils import *

show_result = 'SHOW'
algorithm = 'loss_aug_esf'
nb_samples = 10

loss_stddev = 10
loss_scalar = 1
N = 35

np.random.seed(0)

workspaces = load_workspace_dataset(basename='sdf_data_100_' +
                                             str(nb_samples) + '.hdf5')
costs = workspaces[0].costmap
starts = workspaces[0].starts
targets = workspaces[0].targets
demonstrations = workspaces[0].demonstrations
occupancy = workspaces[0].occupancy

box = EnvBox()
lims = workspaces[0].lims
box.dim[0] = lims[0][1] - lims[0][0]
box.dim[1] = lims[1][1] - lims[1][0]
box.origin[0] = box.dim[0] / 2.
box.origin[1] = box.dim[1] / 2.
workspace_box = Workspace(box)

map = np.ones(costs.shape)
if algorithm == 'learch':
    map, op = get_learch_target(map, demonstrations, starts,
                                targets, loss_stddev, loss_scalar,
                                workspace_box)
elif algorithm == 'maxEnt':
    map, f_expected, f_empirical = get_maxEnt_target(map, N, demonstrations, starts,
                                                     targets, workspace_box)
elif algorithm == 'occ':
    map = get_esf_target(map, N, demonstrations, starts, targets,
                         workspace_box)
elif algorithm == 'loss_aug_esf':
    map = get_loss_aug_esf_target(map, demonstrations, starts,
                                  targets, loss_scalar, loss_stddev,
                                  N, workspace_box)
elif algorithm == 'avg_learch_esf':
    map = get_avg_learch_esf_target()

# one_map = get_costmap(phi, np.ones(nb_rbfs ** 2))
print(occupancy.shape)
print(map.shape)
print(costs.shape)
# Output costmap
# show_multiple([occupancy, map], [costs], workspace_box, show_result
# ,starts=starts, targets=targets, paths=paths#, optimal_paths=[op]* 2
#              )
# show(occupancy, workspace_box, show_result,
#     # ,starts=starts, targets=targets, paths=paths, optimal_paths=[op]
#     directory = home + '/../results/figures/occupancy.pdf')
show(map, workspace_box, show_result,
     # ,starts=starts, targets=targets, paths=paths, optimal_paths=[op]
     directory=home + '/../results/figures/nn_target_loss_aug_esf_map.pdf')
show(costs, workspace_box, show_result
     , starts=starts, targets=targets, paths=demonstrations,  # optimal_paths=op,
     directory=home + '/../results/figures/nn_costs_loss_aug_esf.pdf'
     )
