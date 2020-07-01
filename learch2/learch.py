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

import time
import numpy as np
from pyrieef.geometry.interpolation import *
from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *
from pyrieef.learning.inverse_optimal_control import *
from scipy.interpolate import Rbf


class Learch2D(Learch):
    """ Implements the LEARCH algorithm for a 2D squared map """

    def __init__(self, nb_points, centers, sigma,
                 paths, starts, targets, workspace):
        Learch.__init__(self, len(paths))

        self.workspace = workspace

        # Parameters to compute the loss map
        self._loss_scalar = .8
        self._loss_stddev = .8
        # Parameters to compute the step size
        self._learning_rate = 0.2
        self._stepsize_scalar = 1

        # Parameters to compute the cost map from RBFs
        self.nb_points = nb_points
        self.centers = centers
        self.sigma = sigma
        self.phi = get_phi(nb_points, centers, sigma, workspace)

        # Examples
        self.sample_trajectories = paths
        self.sample_starts = starts
        self.sample_targets = targets

        self.w = np.ones(len(centers))
        self.costmap = np.zeros((nb_points, nb_points))
        self.loss_map = np.zeros((len(paths), nb_points, nb_points))
        self.D = np.empty((3, 0))

        # Data structures to save the progress of LEARCH in each iteration
        self.optimal_paths = []
        self.maps = []

        self.initialize_mydata()

    def initialize_mydata(self):
        """ Create the loss maps for each sample trajectory
            initialize the costmap with RBFs each of weight 1
        """
        for i, t in enumerate(self.sample_trajectories):
            self.loss_map[i] = scaled_hamming_loss_map(
                t, self.nb_points, self._loss_scalar, self._loss_stddev)
        self.costmap = get_costmap(
            self.nb_points, self.centers, self.sigma, self.w, self.workspace)

    def planning(self):
        """ Compute the optimal path for each start and
            target state in the expample trajectories
            Add the states to self.D where the cost function has to
            increase/decrease
        """
        # data structure saving the optimal trajectories in the current costmap
        op = [None] * len(self.sample_trajectories)

        converter = CostmapToSparseGraph(self.costmap)
        converter.integral_cost = True
        graph = converter.convert()
        pixel_map = self.workspace.pixel_map(self.nb_points)

        for i, trajectory in enumerate(self.sample_trajectories):
            # Get start and target state
            s = pixel_map.world_to_grid(self.sample_starts[i])
            t = pixel_map.world_to_grid(self.sample_targets[i])

            try:
                # Compute the shortest path between the start and the target
                optimal_path = converter.dijkstra_on_map(
                    np.exp(self.costmap - self.loss_map[i]),
                    s[0], s[1], t[0], t[1])
                op[i] = optimal_path
            except Exception as e:
                print("Exception")
                continue

            # Add the states of the optimal trajectory to D
            # The costs should be increased in the states
            # of the optimal trajectory
            x1 = np.asarray(np.transpose(optimal_path)[:][0])
            x2 = np.asarray(np.transpose(optimal_path)[:][1])
            D = np.vstack((x1, x2, np.ones(x1.shape)))
            self.D = np.hstack((self.D, D))

            # Add the states of the example trajectory to D
            # The costs should be decreased in the states
            # of the example trajectory
            x1 = np.asarray(np.transpose(trajectory)[:][0])
            x2 = np.asarray(np.transpose(trajectory)[:][1])
            D = np.vstack((x1, x2, - np.ones(x1.shape)))
            self.D = np.hstack((self.D, D))
        self.optimal_paths.append(op)

    def supervised_learning(self, t):
        """ Train a regressor on D
            compute the new weights with gradient descent
        """
        C = self.D[:][2]
        x1 = self.D[:][0].astype(int)
        x2 = self.D[:][1].astype(int)

        Phi = self.phi[:, x1, x2].T
        w_new = linear_regression(Phi, C, self.w, 6, 0)
        self.w = self.w + self.get_stepsize(t) * w_new
        self.costmap = get_costmap(
            self.nb_points, self.centers, self.sigma, self.w, self.workspace)

    def one_step(self, t):
        """ Compute one step of the LEARCH algorithm """
        time_0 = time.time()
        print("step :", t)
        self.planning()
        self.supervised_learning(t)
        print("took t : {} sec.".format(time.time() - time_0))
        return self.maps, self.optimal_paths

    def n_step(self, n):
        """ Compute n steps of the LEARCH algorithm """
        for i in range(n):
            self.one_step(i)
            self.maps.append(self.costmap)
        return self.maps, self.optimal_paths

    def solve(self):
        """ Compute LEARCH until the weights converge """
        w_old = copy.deepcopy(self.w)
        e = 10
        i = 0
        while e > 1:
            self.one_step(i)
            print("w: ", self.w)
            self.maps.append(self.costmap)
            e = (np.absolute(self.w - w_old)).sum()
            print("convergence: ", e)
            w_old = copy.deepcopy(self.w)
            i += 1
        return self.maps, self.optimal_paths

    def get_stepsize(self, t):
        """ Returns the step size for gradient descent
            alpha = r / (t + m)
                r: learning rate
                t: iteration
                m: scalar (specifies where the steps get more narrow)
        """
        return self._learning_rate / (t + self._stepsize_scalar)

    def get_squared_stepsize(self, t):
        """ Returns the step size for gradient descent
            alpha = r / sqrt(t + m)
                r: learning rate
                t: iteration
                m: scalar (specifies where the steps get more narrow)
        """
        return self._learning_rate / np.sqrt(t + self._stepsize_scalar)


def scaled_hamming_loss_map(trajectory, nb_points,
                            goodness_scalar, goodness_stddev):
    """ Create a map from a given trajectory with a scaled hamming loss
        with small values near by the trajectory
        and larger values further away from the trajectory
    """
    occpancy_map = np.zeros((nb_points, nb_points))
    for x_1, x_2 in trajectory:
        occpancy_map[x_1][x_2] = 1
    goodness = goodness_scalar * np.exp(-0.5 * (
            edt(occpancy_map) / goodness_stddev) ** 2)
    return 1 - goodness


def hamming_loss_map(trajectory, nb_points):
    """ Create a map from a given trajectory with the hamming loss
        with 0 in all states of the given trajectory
        and 1 everywhere else
    """
    occpancy_map = np.ones((nb_points, nb_points))
    for x_1, x_2 in trajectory:
        occpancy_map[x_1][x_2] = 0
    return occpancy_map


def get_rbf(nb_points, center, sigma, workspace):
    """ Returns a radial basis function phi_i as map
        phi_i = exp(-(x-center/sigma)**2)
    """
    X, Y = workspace.box.meshgrid(nb_points)
    rbf = Rbf(center[0], center[1], 1, function='gaussian', epsilon=sigma)
    map = rbf(X, Y)

    return map


def get_phi(nb_points, centers, sigma, workspace):
    """ Returns the radial basis functions as vector """
    rbfs = []
    for i, center in enumerate(centers):
        rbfs.append(get_rbf(nb_points, center, sigma, workspace))
    phi = np.stack(rbfs)

    return phi


def get_costmap(nb_points, centers, sigma, w, workspace):
    """ Returns the costmap of RBFs"""
    costmap = np.tensordot(w,
                           get_phi(nb_points, centers, sigma, workspace),
                           axes=1)
    return costmap


def plan_paths(nb_samples, costmap, workspace, average_cost=False):
    # Plan example trajectories
    converter = CostmapToSparseGraph(costmap, average_cost)
    converter.integral_cost = True
    graph = converter.convert()
    pixel_map = workspace.pixel_map(costmap.shape[0])

    paths = []
    starts = []
    targets = []
    for i in range(nb_samples):
        # Choose start and target of the trajectory randomly
        s_w = sample_collision_free(workspace)
        t_w = sample_collision_free(workspace)
        s = pixel_map.world_to_grid(s_w)
        t = pixel_map.world_to_grid(t_w)
        try:
            print("planning...")
            time_0 = time.time()
            # Compute the shortest path between the start and the target
            path = converter.dijkstra_on_map(
                np.exp(costmap), s[0], s[1], t[0], t[1])
        except Exception as e:
            print("Exception")

        paths.append(path)
        starts.append(s_w)
        targets.append(t_w)
        print("took t : {} sec.".format(time.time() - time_0))

    return starts, targets, paths
