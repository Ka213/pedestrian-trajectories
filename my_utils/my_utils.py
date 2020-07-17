from common_import import *

import time
import numpy as np
from pyrieef.geometry.workspace import *
from pyrieef.geometry.interpolation import *
from pyrieef.graph.shortest_path import *


def get_edt(optimal_trajectory, sample_trajectory, nb_points):
    """ Return the euclidean distance transform
        of the optimal trajectory to the example trajectory
    """
    occpancy_map = np.zeros((nb_points, nb_points))
    x_1 = np.asarray(sample_trajectory)[:, 0]
    x_2 = np.asarray(sample_trajectory)[:, 1]
    occpancy_map[x_1, x_2] = 1
    distance = edt(occpancy_map)
    return np.sum(distance[np.asarray(optimal_trajectory).astype(int)])


def get_expected_edge_frequency(transition_probability, costmap, N, nb_points):
    """ Set the visitation frequency """
    Z_s = np.ones(nb_points ** 2)
    Z_a = np.multiply(np.dot(transition_probability, Z_s.T).
                      reshape(nb_points ** 2, 8),
                      np.tile(np.exp(costmap.reshape(nb_points ** 2)),
                              (8, 1)).T)
    Z_s = np.sum(Z_a, axis=1)
    P = Z_a / np.tile(Z_s, (8, 1)).T
    D = np.ones((nb_points ** 2, N + 1)) / nb_points ** 2
    for t in range(1, N):
        D[:, t + 1] = np.sum(P * np.dot(transition_probability, D[:, t].T).
                             reshape(nb_points ** 2, 8), axis=1)
    visitation_frequency = np.sum(D, axis=1)
    return visitation_frequency


def get_stepsize(t, learning_rate, stepsize_scalar):
    """ Returns the step size for gradient descent
        alpha = r / (t + m)
            r: learning rate
            t: iteration
            m: scalar (specifies where the steps get more narrow)
    """
    return learning_rate / (t + stepsize_scalar)


def get_squared_stepsize(t, learning_rate, stepsize_scalar):
    """ Returns the step size for gradient descent
        alpha = r / sqrt(t + m)
            r: learning rate
            t: iteration
            m: scalar (specifies where the steps get more narrow)
    """
    return learning_rate / np.sqrt(t + stepsize_scalar)


def scaled_hamming_loss_map(trajectory, nb_points,
                            goodness_scalar, goodness_stddev):
    """ Create a map from a given trajectory with a scaled hamming loss
        with small values near by the trajectory
        and larger values further away from the trajectory
    """
    occpancy_map = np.zeros((nb_points, nb_points))
    x_1 = np.asarray(trajectory)[:, 0]
    x_2 = np.asarray(trajectory)[:, 1]
    occpancy_map[x_1, x_2] = 1
    goodness = goodness_scalar * np.exp(-0.5 * (
            edt(occpancy_map) / goodness_stddev) ** 2)
    return 1 - goodness


def hamming_loss_map(trajectory, nb_points):
    """ Create a map from a given trajectory with the hamming loss
        with 0 in all states of the given trajectory
        and 1 everywhere else
    """
    occpancy_map = np.ones((nb_points, nb_points))
    x_1 = np.asarray(trajectory)[:, 0]
    x_2 = np.asarray(trajectory)[:, 1]
    occpancy_map[x_1][x_2] = 0
    return occpancy_map


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
            path = converter.dijkstra_on_map(costmap, s[0], s[1], t[0], t[1])
        except Exception as e:
            print("Exception")

        paths.append(path)
        starts.append(s_w)
        targets.append(t_w)
        print("took t : {} sec.".format(time.time() - time_0))

    return starts, targets, paths
