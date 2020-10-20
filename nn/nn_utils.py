from common_import import *

from my_utils.my_utils import *
from my_utils.environment import *
from pyrieef.geometry.workspace import *


def get_learch_input(costmap, paths, starts, targets, loss_stddev, loss_scalar):
    map = np.zeros(costmap.shape)

    # Push down on demonstrations
    for i, trajectory in enumerate(paths):
        map -= hamming_loss_map(trajectory, costmap.shape[0])

    # Push up on optimal path
    op = []
    for i, (path, start, target) in enumerate(zip(paths, starts, targets)):
        c = costmap - scaled_hamming_loss_map(path, costmap.shape[0],
                                              loss_stddev, loss_scalar)
        c = c - c.min()
        _, _, trajectory = plan_paths(1, c, Workspace(),
                                      starts=[start],
                                      targets=[target])
        op.append(trajectory[0])
        map += hamming_loss_map(trajectory[0], costmap.shape[0])
    return map, op


def get_maxEnt_input(costmap, N, paths, targets, phi):
    try:  # TODO on log costmap?
        D = get_expected_edge_frequency(costmap, N, costmap.shape[0], targets,
                                        paths, Workspace())
    except Exception:
        raise
    f_expected = np.tensordot(phi, D)
    f_empirical = get_empirical_feature_count(paths, phi)
    f = f_empirical - f_expected
    f = - f - np.min(- f)
    map = get_costmap(phi, f)
    return map


def get_occ_input(costmap, N, paths, targets):
    # Push up on state frequency
    occ = get_expected_edge_frequency(costmap, N, costmap.shape[0], targets,
                                      paths, Workspace())
    # Push down on demonstrations
    map = np.zeros(costmap.shape)
    for i, trajectory in enumerate(paths):
        map -= hamming_loss_map(trajectory, costmap.shape[0])
    # Scaleing
    occ = occ / occ.sum() * - map.sum()
    map = occ + map
    return map


def get_loss_aug_occ_input(costmap, paths, targets, loss_scalar, loss_stddev, N):
    loss_augmentation = np.zeros(costmap.shape)
    map = np.zeros(costmap.shape)
    # Push down on demonstrations
    for i, trajectory in enumerate(paths):
        map -= hamming_loss_map(trajectory, costmap.shape[0])
        loss_augmentation += scaled_hamming_loss_map(trajectory,
                                                     costmap.shape[0],
                                                     loss_scalar /
                                                     len(paths),
                                                     loss_stddev)
    # Push up on loss augmented state frequency
    occ = get_expected_edge_frequency(costmap - loss_augmentation, N,
                                      costmap.shape[0], targets, paths,
                                      Workspace())
    # Scaleing
    occ = occ / occ.sum() * - map.sum()
    map = occ + map
    return map


def get_avg_learch_occ_input():
    # TODO implement
    print("please implement")
