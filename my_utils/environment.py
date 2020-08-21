import common_import

import time
import numpy as np
from scipy.interpolate import Rbf

from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *


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


def create_random_environment(nb_points, nb_rbfs, sigma, nb_samples, workspace):
    """ Returns a random environment """
    # Create costmap with rbfs
    w = np.random.random(nb_rbfs ** 2)
    centers = workspace.box.meshgrid_points(nb_rbfs)
    costmap = get_costmap(nb_points, centers, sigma, w, workspace)

    # Plan example trajectories
    starts, targets, paths = plan_paths(nb_samples, costmap, workspace)

    return w, costmap, starts, targets, paths


def plan_paths(nb_samples, costmap, workspace, starts=None, targets=None,
               average_cost=False):
    """ Plan example trajectories
        either with random or fixed start and target state
    """
    converter = CostmapToSparseGraph(costmap, average_cost)
    converter.integral_cost = True
    graph = converter.convert()
    pixel_map = workspace.pixel_map(costmap.shape[0])

    paths = []
    # Choose starts of the trajectory randomly
    if starts is None:
        starts = []
        for i in range(nb_samples):
            s_w = sample_collision_free(workspace)
            starts.append(s_w)
    # Choose targets of the trajectory randomly
    if targets is None:
        targets = []
        for i in range(nb_samples):
            t_w = sample_collision_free(workspace)
            targets.append(t_w)
    # Plan path
    for s_w, t_w in zip(starts, targets):
        s = pixel_map.world_to_grid(s_w)
        t = pixel_map.world_to_grid(t_w)
        try:
            # print("planning...")
            time_0 = time.time()
            # Compute the shortest path between the start and the target
            path = converter.dijkstra_on_map(costmap, s[0], s[1], t[0], t[1])
            paths.append(path)
        except Exception as e:
            print("Exception while planning a path")
            print(e)
            paths.append([(s[0], s[1])])
            continue
        # print("took t : {} sec.".format(time.time() - time_0))

    return starts, targets, paths
