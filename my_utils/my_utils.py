from common_import import *

import time
import warnings
import numpy as np
from sklearn.metrics import log_loss
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


def get_empirical_feature_count(sample_trajectories, Phi):
    """ Return the expected empirical feature counts """
    f = np.zeros(Phi.shape[0])
    for i, trajectory in enumerate(sample_trajectories):
        x_1 = np.asarray(trajectory)[:, 0]
        x_2 = np.asarray(trajectory)[:, 1]
        f += np.sum(Phi[:, x_1.astype(int), x_2.astype(int)],
                    axis=1)
    f = f / len(sample_trajectories)
    return f


def get_expected_edge_frequency(transition_probability, costmap, N, nb_points,
                                terminal_states, paths, workspace):
    """ Return the expected state visitation frequency """
    converter = CostmapToSparseGraph(costmap)
    converter.integral_cost = True
    graph = converter.convert()

    terminal = []
    pixel_map = workspace.pixel_map(nb_points)
    for t in terminal_states:
        t = pixel_map.world_to_grid(t)
        terminal.append(converter.graph_id(t[0], t[1]))

    # Backward pass
    Z_s = np.zeros((nb_points ** 2))
    Z_s[terminal] = 1
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            for i in range(N):
                Z_a = np.multiply(np.dot(transition_probability, Z_s.T).
                                  reshape(nb_points ** 2, 8), np.tile(np.exp(
                    -costmap.reshape(nb_points ** 2)), (8, 1)).T)
                Z_s = np.sum(Z_a, axis=1)
            # Local Action Probability Computation
            P = Z_a / np.tile(Z_s, (8, 1)).T
            # Forward Pass
            D = np.zeros((nb_points ** 2, N + 1))
            # Initial state probabilities
            l = 0
            for path in paths:
                for point in path:
                    x = converter.graph_id(point[0], point[1])
                    l += 1
                    D[x, 0] = 1
            D[:, 0] = D[:, 0] / l
            for t in range(0, N):
                D[:, t + 1] = np.sum(P * np.dot(transition_probability, D[:, t]).
                                     reshape(nb_points ** 2, 8), axis=1)
            # Summing frequencies
            visitation_frequency = np.sum(D, axis=1).reshape((nb_points, nb_points))
            return visitation_frequency.T
        except Warning as w:
            print("Warning happend while computing the expected edge frequency")
            print(w)
            raise
            return np.zeros((nb_points, nb_points))


def get_policy(costmap):
    ''' Generate policy from costmap
        returns array with shape (nb_points ** 2)
        go from state policy[i] to state i
    '''
    converter = CostmapToSparseGraph(np.exp(costmap))
    converter.integral_cost = True
    graph = converter.convert()

    predecessors = shortest_paths(converter._graph_dense)
    policy = np.zeros(costmap.shape[0] ** 2)
    for i, p in enumerate(predecessors.T):
        policy[i] = np.bincount(np.abs(p)).argmax()
    return policy


def policy_iteration(costmap, nb_points, discount,
                     transition_probability):
    """ Compute policy iteration on the given costmap """
    Q = np.zeros((nb_points ** 2, 8))
    Q_old = copy.deepcopy(Q)
    e = 10
    while e > 1:
        Q = np.tile(costmap, (8, 1)).reshape((nb_points ** 2, 8)) + discount * \
            np.dot(transition_probability, np.amax(Q, axis=1).T) \
                .reshape((nb_points ** 2, 8))
        e = np.amax(np.abs(Q - Q_old))
        Q_old = copy.deepcopy(Q)

    return Q


def get_transition_probabilities(costmap, nb_points):
    """ Set transition probability matrix """
    transition_probability = np.zeros((nb_points ** 2 * 8,
                                       nb_points ** 2))
    converter = CostmapToSparseGraph(costmap)
    converter.integral_cost = True
    graph = converter.convert()

    for i in range(nb_points ** 2):
        # Get neighbouring states in the order
        # down, up, right, down-right, up-right, left, down-left, up-left
        s = converter.costmap_id(i)
        for j, n in enumerate(converter.neiborghs(s[0], s[1])):
            if converter.is_in_costmap(n[0], n[1]):
                x = converter.graph_id(n[0], n[1])
                transition_probability[i * 8 + j, x] = 1
    return transition_probability


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
    return goodness_scalar - goodness


def hamming_loss_map(trajectory, nb_points):
    """ Create a map from a given trajectory with the hamming loss
        with 0 in all states of the given trajectory
        and 1 everywhere else
    """
    occpancy_map = np.ones((nb_points, nb_points))
    x_1 = np.asarray(trajectory)[:, 0]
    x_2 = np.asarray(trajectory)[:, 1]
    occpancy_map[x_1, x_2] = 0
    return occpancy_map


def get_edt_loss(nb_points, learned_paths, demonstrations, nb_samples):
    """ Return the loss of the euclidean distance transform
        between the demonstrations and learned paths
    """
    loss = 0
    for op, d in zip(learned_paths, demonstrations):
        loss += get_edt(op, d, nb_points) / len(d)
    loss = loss / nb_samples
    return loss


def get_overall_loss(nb_points, learned_map, original_costmap):
    """ Return the difference between the costsmaps """
    loss = np.sum(np.abs(learned_map - original_costmap)) / (nb_points ** 2)
    return loss


def get_nll(learned_paths, demonstrations, nb_points):
    """ Return the negative log likelihood
        of the learned paths and the predictions
    """
    loss = 0
    for l, d in zip(learned_paths, demonstrations):
        pred = hamming_loss_map(l, nb_points)
        truth = hamming_loss_map(d, nb_points)
        loss += log_loss(truth, pred)
    return loss / len(learned_paths)
