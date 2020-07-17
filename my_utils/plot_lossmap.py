from common_import import *

import numpy as np
from pyrieef.geometry.workspace import Workspace
from my_utils import *
from output import *

show_result = 'SHOW'
average_cost = False
nb_points = 40
loss_scalar = 1
loss_stddev = 1

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)
map = np.ones((nb_points, nb_points))

# Plan trajectory
starts, targets, paths = plan_paths(1, map, workspace)

# Create lossmap
lossmap = scaled_hamming_loss_map(paths[0],
                                  nb_points, loss_scalar, loss_stddev)

# Output lossmap
title = 'scaled hamming loss map; loss_scalar: {}, ' \
        'loss_stddev: {}'.format(loss_scalar, loss_stddev)
path = home + '/../figures/lossmap_{}.pdf'.format(4)
show(lossmap, workspace, show_result, directory=path, title=title)
