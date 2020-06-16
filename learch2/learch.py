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

import numpy as np
from pyrieef.geometry.interpolation import linear_regression
from pyrieef.graph.shortest_path import *
from pyrieef.learning.inverse_optimal_control import *

class Learch2D(Learch):

    def __init__(self, costmap, paths, starts, targets, pixel_map):
        Learch.__init__(self, len(paths))
        a, b = costmap.shape
        c = len(paths)
        self._nb_points = a
        self._goodness_scalar = .5
        self._goodness_stddev = .8
        self.costmap = costmap
        self.sample_trajectories = paths
        self.sample_starts = starts
        self.sample_targets = targets

        self.goodness_map = np.zeros((c, a,b))
        self.d = []
        self.pixel_map = pixel_map
        self.initialize_mydata()

    def initialize_mydata(self):
        for i, t in enumerate(self.sample_trajectories):
            self.goodness_map[i] = my_goodness_map(t,self._nb_points , self._goodness_scalar, self._goodness_stddev)

    def planning(self):
        self.d = []
        i = 0
        for trajectory in self.sample_trajectories:

            converter = CostmapToSparseGraph(self.costmap)
            converter.integral_cost = True
            graph = converter.convert()

            s = self.pixel_map.world_to_grid(self.sample_starts[i])
            t = self.pixel_map.world_to_grid(self.sample_targets[i])

            try:
                optimal_path = converter.dijkstra_on_map(np.exp(self.costmap - self.goodness_map[i]), s[0], s[1], t[0], t[1])
            except Exception as e:
                continue
            gradient = 0.2 # TODO compute gradient
            for x_1,x_2 in optimal_path:
                self.d.append([x_1, x_2, self.costmap[x_1][x_2] + gradient])
                self.costmap[x_1][x_2] = self.costmap[x_1][x_2] + gradient
            for x_1,x_2 in trajectory:
                self.costmap[x_1][x_2] = self.costmap[x_1][x_2] - gradient
                self.d.append([x_1, x_2, self.costmap[x_1][x_2] - gradient])
            i = i + 1

    def supervised_learning(self):

        X = np.zeros((2,len(self.d)))
        X[0] = np.asarray(np.transpose(self.d)[:][0])
        X[1] = np.asarray(np.transpose(self.d)[:][1])
        X = np.transpose(X)

        Y = np.asarray(np.transpose(self.d)[:][2])
        w = np.ones(X.shape[1])

        #TODO find best lambdas
        beta = linear_regression(X,Y, w, 0.2, 0.2)
        for i, [x_1, x_2] in enumerate(X):
            self.costmap[int(x_1)][int(x_2)] = np.dot(X[i], beta)

    def one_step(self):
        self.planning()
        self.supervised_learning()

    def solve(self):
        old_c = self.costmap
        while e > 0.1:
            self.planning()
            self.supervised_learning()
            e = (np.absolute(self.costmap-old_c)).sum()
            old_c = self.costmap


def my_goodness_map(trajectory, nb_points,
                 goodness_scalar,
                 goodness_stddev):
    occpancy_map = np.zeros((nb_points, nb_points))
    for x_1, x_2 in trajectory:
        occpancy_map[x_1][x_2] = 1
    return goodness_scalar * np.exp(-0.5 * (
        edt(occpancy_map) / goodness_stddev)**2)