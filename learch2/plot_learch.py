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

from pyrieef.geometry.interpolation import *
from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *
from pyrieef.learning.inverse_optimal_control import *
from learch2.learch import *
from learch2.output import *

show_result = True
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 8
nb_steps = 10

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)

# Create costmap with rbfs
w = np.random.random(nb_rbfs ** 2)
centers = workspace.box.meshgrid_points(nb_rbfs)
original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)

# Plan example trajectories
starts, targets, paths = plan_paths(nb_samples, original_costmap, workspace)

# Learn costmap
l = Learch2D(nb_points, centers, sigma, paths, starts, targets, workspace)
maps, optimal_paths = l.n_step(nb_steps)

# Output learned costmaps
show_multiple_maps(maps, workspace, starts=starts, targets=targets,
                   paths=paths, optimal_paths=optimal_paths)
show_multiple_maps([original_costmap, maps[-1]], workspace)
