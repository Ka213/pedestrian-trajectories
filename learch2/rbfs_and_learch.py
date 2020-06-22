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
import sys

from scipy.interpolate import Rbf

sys.path.insert(0,"C:/Users/Katharina/Documents/Studium/10Semester/Masterarbeit/DeepLearch/pyrieef")
from pyrieef.learning.inverse_optimal_control import *
from pyrieef.geometry.differentiable_geometry import RadialBasisFunction
from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *
import pyrieef.rendering.workspace_renderer as render
import numpy as np
import time
import matplotlib.pyplot as plt

class Learch2D(Learch):

    def __init__(self, costmap, paths, starts, targets, pixel_map):
        Learch.__init__(self, len(paths))

        self._nb_points = costmap.shape[0]
        self._loss_scalar = .8
        self._loss_stddev = .8
        self.costmap = costmap
        self.sample_trajectories = paths
        self.sample_starts = starts
        self.sample_targets = targets

        self.loss_map = np.zeros((len(paths), costmap.shape[0], costmap.shape[1]))
        self.d = []
        self.pixel_map = pixel_map
        self.initialize_mydata()

    def initialize_mydata(self):
        for i, t in enumerate(self.sample_trajectories):
            self.loss_map[i] = scaled_hamming_loss_map(t, self._nb_points, self._loss_scalar, self._loss_stddev)

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
                optimal_path = converter.dijkstra_on_map(np.exp(self.costmap - self.loss_map[i]), s[0], s[1], t[0], t[1])
            except Exception as e:
                print("Exception")
                continue
            g = get_gradient(self.costmap)
            for x_1,x_2 in optimal_path:
                self.d.append([x_1, x_2, self.costmap[x_1][x_2] + g[x_1][x_2]])
            for x_1,x_2 in trajectory:
                self.d.append([x_1, x_2, self.costmap[x_1][x_2] - g[x_1][x_2]])
            i = i + 1

    def supervised_learning(self, w_t):
        C = np.asarray(np.transpose(self.d)[:][2])
        phi = np.zeros((len(centers), nb_points, nb_points))
        for j in range(len(centers)):
            phi[j] = get_rbf(nb_points, centers[j], sigma, w_t[j])
        Phi = np.ones((len(self.d), nb_rbfs**2))
        for i, d in enumerate(self.d):
            x_1 = d[0]
            x_2 = d[1]
            for j in range(len(centers)):
                Phi[i][j] = phi[j][x_1][x_2]
        w = linear_regression(Phi, C, w_t, 0.2, 0)
        return w

    def one_step(self,w_t):
        self.planning()
        w = self.supervised_learning(w_t)
        return w

    def solve(self, w):
        old_c = copy.deepcopy(self.costmap)
        e_p = 1
        e_o = 1
        while e_p < 60:
            w = self.one_step(w)
            print("w: ", w)
            self.costmap = get_costmap(nb_points, centers, sigma, w)
            maps.append(self.costmap)
            #show_map(self.costmap, workspace)
            e_p = e_p + 1 #(np.absolute(self.costmap-old_c)).sum()
            e_o = (np.absolute(self.costmap-original_costmap)).sum()
            print("error with previous: ", e_p)
            print("error with original: ", e_o)
            old_c = copy.deepcopy(self.costmap)
        show_multiple_maps(maps, workspace)


def scaled_hamming_loss_map(trajectory, nb_points,
                 goodness_scalar,
                 goodness_stddev):
    occpancy_map = np.zeros((nb_points, nb_points))
    for x_1, x_2 in trajectory:
        occpancy_map[x_1][x_2] = 1
    goodness = goodness_scalar * np.exp(-0.5 * (
        edt(occpancy_map) / goodness_stddev)**2)
    return 1-goodness


def hamming_loss_map(trajectory, nb_points,
                 goodness_scalar,
                 goodness_stddev):
    occpancy_map = np.ones((nb_points, nb_points))
    for x_1, x_2 in trajectory:
        occpancy_map[x_1][x_2] = 0
    print(occpancy_map[1][0])
    return occpancy_map

def get_gradient(costmap):
    g = np.ones(costmap.shape)
    #g = np.gradient(costmap)
    # gradient fixed for now
    return  g * 0.2 #g[o]

def show_map(costmap, workspace):
    if show_result:
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
        viewer.draw_ws_img(costmap, interpolate="none")
        viewer.show_once()

def show_samples(costmap, starts, targets, paths, workspace):
    if show_result:
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
        viewer.draw_ws_img(costmap, interpolate="none")
        for s_w, t_w, path in zip(starts, targets, paths):
            trajectory = [None] * len(path)
            for i, p in enumerate(path):
                trajectory[i] = pixel_map.grid_to_world(np.array(p))
            c = cmap(np.random.rand())
            viewer.draw_ws_line(trajectory, color=c)
            viewer.draw_ws_point(s_w, color=c)
            viewer.draw_ws_point(t_w, color=c)
        viewer.show_once()

def show_multiple_maps(costmaps, workspace):
    if show_result:
        r = math.ceil(math.sqrt(len(costmaps)))
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True, rows=r, cols=r)
        for i, costmap in enumerate(costmaps):
            viewer.set_drawing_axis(i)
            viewer.draw_ws_img(costmap, interpolate="none")
        viewer.show_once()

def get_costmap(nb_points, centers, sigma, w):
    for i, x0 in enumerate(centers):
        rbf[i] = Scale(RadialBasisFunction(x0, sigma * np.eye(2)), w[i])
    phi = SumOfTerms(rbf)
    X, Y = workspace.box.meshgrid(nb_points)
    costmap = two_dimension_function_evaluation(X, Y, phi)

    return costmap

def get_rbf(nb_points, center, sigma, w):
    X, Y = workspace.box.meshgrid(nb_points)
    scipy_rbf = Rbf(center[0], center[1], w, function='gaussian', epsilon=sigma, norm='minkowski', p=10)
    grid_rbf = scipy_rbf(X, Y)

    return grid_rbf

def sum_rbfs(nb_points, centers, sigma, w):
    costmap = np.zeros((nb_points, nb_points))
    for i, center in enumerate(centers):
        grid_rbf = get_rbf(nb_points, center, sigma, w[i])
        costmap = costmap + grid_rbf

    return costmap

cmap = plt.get_cmap('viridis')

show_result = True
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 200
nb_samples = 10

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)

#create costmap with rbfs
w = np.random.random(nb_rbfs**2)
print("weights: ", w)
rbf = [None] * nb_rbfs**2
centers = workspace.box.meshgrid_points(nb_rbfs)

original_costmap = sum_rbfs(nb_points, centers, sigma, w)
maps = [original_costmap]

single_rbf = np.zeros((nb_rbfs**2, nb_points, nb_points))
for i, c in enumerate(centers):
    single_rbf[i] = get_rbf(nb_points, c, sigma, w[i])
show_multiple_maps(single_rbf,workspace)
sum = sum_rbfs(nb_points, centers, sigma, w)
show_map(sum, workspace)

# Plan path
converter = CostmapToSparseGraph(original_costmap, average_cost)
converter.integral_cost = True
graph = converter.convert()

paths = []
starts = []
targets = []
for i in range(nb_samples):
    s_w = sample_collision_free(workspace)
    t_w = sample_collision_free(workspace)
    s = pixel_map.world_to_grid(s_w)
    t = pixel_map.world_to_grid(t_w)
    try:
        print("planning...")
        time_0 = time.time()
        path = converter.dijkstra_on_map(np.exp(original_costmap), s[0], s[1], t[0], t[1])
    except Exception as e:
        print("Exception")

    paths.append(path)
    starts.append(s_w)
    targets.append(t_w)
    print("took t : {} sec.".format(time.time() - time_0))

show_samples(original_costmap, starts, targets, paths, workspace)

#learn costs
w_t = np.ones(nb_rbfs**2)
costmap = get_costmap(nb_points, centers, sigma, w_t)
maps.append(costmap)
show_map(costmap, workspace)

l = Learch2D(costmap, paths, starts, targets, pixel_map)
l.solve(w_t)