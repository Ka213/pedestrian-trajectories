#!/usr/bin/env python

# Copyright (c) 2018, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                        Jim Mainprice on Sunday June 13 2018
import common_import

import numpy as np
from pyrieef.geometry.workspace import Workspace
from learch2.learch import scaled_hamming_loss_map, plan_paths
from learch2.output import show_map, save_map

show_result = True
average_cost = False
nb_points = 40
loss_scalar = 0.8
loss_stddev = 0.8

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
path = '../figures/lossmap_{}.png'.format(2)
save_map(lossmap, workspace, directory=path, title=title)
show_map(lossmap, workspace, title=title)
