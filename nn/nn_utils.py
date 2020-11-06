from common_import import *

from my_utils.my_utils import *
from my_utils.environment import *
from pyrieef.geometry.workspace import *


def get_learch_target(costmap, paths, starts, targets, loss_stddev, loss_scalar):
    map = np.zeros(costmap.shape)

    # Push down on demonstrations
    for i, trajectory in enumerate(paths):
        x1 = np.asarray(trajectory)[:, 0]
        x2 = np.asarray(trajectory)[:, 1]
        map[x1, x2] += costmap[x1, x2] - 1

    # Push up on optimal path
    op = []
    for i, (path, start, target) in enumerate(zip(paths, starts, targets)):
        c = costmap - scaled_hamming_loss_map(path, costmap.shape[0],
                                              loss_stddev, loss_scalar)
        c = c - c.min()
        _, _, trajectory = plan_paths(1, c, Workspace(),
                                      starts=[start],
                                      targets=[target])
        trajectory = trajectory[0]
        op.append(trajectory)
        x1 = np.asarray(trajectory)[:, 0]
        x2 = np.asarray(trajectory)[:, 1]
        map[x1, x2] += costmap[x1, x2] + 1
    return map, op


def get_maxEnt_target(costmap, N, paths, starts, targets, phi):
    try:
        D = get_expected_edge_frequency(costmap, N, costmap.shape[0], starts,
                                        targets, Workspace())
    except Exception:
        raise
    f_expected = np.tensordot(phi, D)
    f_empirical = get_empirical_feature_count(paths, phi)
    f = f_empirical - f_expected
    f = - f - np.min(- f)
    map = get_costmap(phi, f)
    return map


def get_esf_target(costmap, N, paths, starts, targets):
    # Push up on state frequency
    occ = get_expected_edge_frequency(costmap, N, costmap.shape[0], starts,
                                      targets, Workspace())
    # Push down on demonstrations
    map = np.zeros(costmap.shape)
    for i, trajectory in enumerate(paths):
        map -= hamming_loss_map(trajectory, costmap.shape[0])
    # Scaleing
    occ = occ / occ.sum() * - map.sum()
    map = occ + map
    return map


def get_loss_aug_est_target(costmap, paths, starts, targets, loss_scalar,
                            loss_stddev, N):
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
                                      costmap.shape[0], starts, targets,
                                      Workspace())
    # Scaleing
    occ = occ / occ.sum() * - map.sum()
    map = occ + map
    return map


def get_avg_learch_esf_target():
    # TODO implement
    print("please implement")
