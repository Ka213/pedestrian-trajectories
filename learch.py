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

from src.pyrieef.pyrieef.learning.inverse_optimal_control import *
from src.pyrieef.pyrieef.geometry.workspace import *
from src.pyrieef.pyrieef.geometry.interpolation import *
from src.pyrieef.pyrieef.graph.shortest_path import *
import src.pyrieef.pyrieef.rendering.workspace_renderer as render
from scipy.interpolate import Rbf
import numpy as np
import time
import matplotlib.pyplot as plt

class Learch2D(Learch):

    def __init__(self, nb_points, centers, sigma, paths, starts, targets, pixel_map):
        Learch.__init__(self, len(paths))

        self._loss_scalar = .8
        self._loss_stddev = .8

        self.nb_points = nb_points
        self.centers = centers
        self.sigma = sigma
        self.phi = get_phi(nb_points, centers, sigma)
        self.sample_trajectories = paths
        self.sample_starts = starts
        self.sample_targets = targets
        self.w = np.ones(len(centers))
        self.costmap = np.zeros((nb_points, nb_points))
        self.loss_map = np.zeros((len(paths), nb_points, nb_points))
        self.d = np.empty((3,0))
        self.pixel_map = pixel_map
        self.initialize_mydata()

    def initialize_mydata(self):
        for i, t in enumerate(self.sample_trajectories):
            self.loss_map[i] = scaled_hamming_loss_map(t, self.nb_points, self._loss_scalar, self._loss_stddev)
        self.costmap = get_costmap(self.nb_points, self.centers, self.sigma, self.w)

    def planning(self):
        for i, trajectory in enumerate(self.sample_trajectories):

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

            x1 = np.asarray(np.transpose(optimal_path)[:][0])
            x2 = np.asarray(np.transpose(optimal_path)[:][1])
            d = np.vstack((x1, x2, self.costmap[x1,x2] + g[x1, x2]))
            self.d = np.hstack((self.d, d))
            x1 = np.asarray(np.transpose(trajectory)[:][0])
            x2 = np.asarray(np.transpose(trajectory)[:][1])
            d = np.vstack(
                (x1, x2, self.costmap[x1, x2] + g[x1, x2]))
            self.d = np.hstack((self.d, d))

    def supervised_learning(self):
        C = self.d[:][2]
        x1 = self.d[:][0].astype(int)
        x2 = self.d[:][1].astype(int)

        Phi = self.phi[:, x1, x2].T
        self.w = linear_regression(Phi, C, self.w, 6, 0)
        self.costmap = get_costmap(self.nb_points, self.centers, self.sigma, self.w)

    def one_step(self):
        self.planning()
        self.supervised_learning()

    def n_step(self, n):
        for i in range(n):
            self.one_step()
            maps.append(self.costmap)
        show_multiple_maps(maps, workspace)

    def solve(self):
        old_c = copy.deepcopy(self.costmap)
        e_p = 1
        e_o = 1
        while e_p < 1:
            self.one_step()
            print("w: ", self.w)
            maps.append(self.costmap)
            e_p = (np.absolute(self.costmap-old_c)).sum()
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
    return occpancy_map

def get_gradient(costmap):
    g = np.ones(costmap.shape)
    #g = np.gradient(costmap)
    # gradient fixed for now
    return  g * 0.5 #g[o]

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
        c = math.ceil(len(costmaps) / r)
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True, rows=r, cols=c)
        for i, costmap in enumerate(costmaps):
            viewer.set_drawing_axis(i)
            viewer.draw_ws_img(costmap, interpolate="none")
        viewer.show_once()

def get_rbf(nb_points, center, sigma):
    X, Y = workspace.box.meshgrid(nb_points)
    rbf = Rbf(center[0], center[1], 1, function='gaussian', epsilon=sigma)
    map = rbf(X, Y)

    return map

def get_phi(nb_points, centers, sigma):
    rbfs = []
    for i, center in enumerate(centers):
        rbfs.append(get_rbf(nb_points, center, sigma))
    phi = np.stack(rbfs)

    return phi

def get_costmap(nb_points, centers, sigma, w):
    costmap = np.tensordot(w, get_phi(nb_points, centers, sigma), axes=1)

    return costmap

cmap = plt.get_cmap('viridis')

show_result = True
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 5

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)

#create costmap with rbfs
w = np.random.random(nb_rbfs**2)
centers = workspace.box.meshgrid_points(nb_rbfs)
phi = get_phi(nb_points, centers, sigma)
original_costmap = get_costmap(nb_points, centers, sigma, w)
maps = [original_costmap]

#show_map(original_costmap, workspace)

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
l = Learch2D(nb_points, centers, sigma, paths, starts, targets, pixel_map)
l.n_step(20)