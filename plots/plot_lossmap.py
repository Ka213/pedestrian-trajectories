from common_import import *

from my_utils.my_utils import *
from my_utils.output_costmap import *
from my_utils.environment import *

show_result = 'SHOW'
average_cost = False
nb_points = 28
loss_scalar = 1
loss_stddev = 10

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)
map = np.ones((nb_points, nb_points))

# Plan trajectory
starts, targets, paths = plan_paths(1, map, workspace)

# Create lossmap
lossmap = scaled_hamming_loss_map(paths[0],
                                  nb_points, loss_scalar, loss_stddev)
print(lossmap.max())
print(lossmap)
# Output lossmap
title = 'scaled hamming loss map; loss_scalar: {}, ' \
        'loss_stddev: {}'.format(loss_scalar, loss_stddev)
path = home + '/../results/figures/lossmap_{}_{}.pdf'.format(loss_scalar,
                                                             loss_stddev)
show(lossmap, workspace, show_result, directory=path, starts=starts,
     targets=targets, paths=paths)
path = home + '/../results/figures/lossmap3D_{}_{}.pdf'.format(loss_scalar,
                                                               loss_stddev)
show_3D(lossmap, workspace, show_result, starts=starts, targets=targets,
        paths=paths, directory=path)
