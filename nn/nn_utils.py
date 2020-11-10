from common_import import *

from my_utils.my_utils import *
from my_utils.environment import *
from pyrieef.geometry.workspace import *


def get_learch_target(costmap, paths, starts, targets, loss_stddev, loss_scalar,
                      workspace):
    map = np.zeros(costmap.shape)
    # Push down on demonstrations
    for i, trajectory in enumerate(paths):
        x1 = np.asarray(trajectory)[:, 0]
        x2 = np.asarray(trajectory)[:, 1]
        map[x1, x2] += - 1  # + costmap[x1, x2]

    # Push up on optimal path
    op = []
    for i, (path, start, target) in enumerate(zip(paths, starts, targets)):
        c = costmap - scaled_hamming_loss_map(path, costmap.shape[0],
                                              loss_scalar, loss_stddev)
        _, _, trajectory = plan_paths(1, c, workspace, starts=[start],
                                      targets=[target])
        trajectory = trajectory[0]
        op.append(trajectory)
        x1 = np.asarray(trajectory)[:, 0]
        x2 = np.asarray(trajectory)[:, 1]
        map[x1, x2] += 1  # + costmap[x1, x2]
    return map, op


def get_maxEnt_target(costmap, N, paths, starts, targets, workspace):
    try:
        d = get_expected_edge_frequency(costmap, N, costmap.shape[0], starts,
                                        targets, workspace)
    except Exception as e:
        print("Exception happened while computing expected state frequencies")
        print(e)
        raise
    phi = np.zeros((costmap.shape[0] ** 2, costmap.shape[0], costmap.shape[1]))
    for i in range(costmap.shape[0]):
        for j in range(costmap.shape[1]):
            phi[i * costmap.shape[0] + j, i, j] = 1
    f_empirical = get_empirical_feature_count(paths, phi)
    f_expected = np.tensordot(phi, d)
    f_empirical = f_empirical / f_empirical.sum()
    f_expected = f_expected / f_expected.sum()
    f = f_empirical - f_expected
    f = - f - np.min(- f)
    map = get_costmap(phi, f)
    return map, f_expected, f_empirical


def get_esf_target(costmap, N, paths, starts, targets, workspace):
    # Push up on state frequency
    occ = get_expected_edge_frequency(costmap, N, costmap.shape[0], starts,
                                      targets, workspace)
    # Push down on demonstrations
    map = np.zeros(costmap.shape)
    for i, trajectory in enumerate(paths):
        map -= hamming_loss_map(trajectory, costmap.shape[0])
    # Scaleing
    occ = occ / occ.sum() * - map.sum()
    map = occ + map
    return map


def get_loss_aug_esf_target(costmap, paths, starts, targets, loss_scalar,
                            loss_stddev, N, workspace):
    map = np.zeros(costmap.shape)
    esf = np.zeros(costmap.shape)
    # Push down on demonstrations
    for i, trajectory in enumerate(paths):
        map -= hamming_loss_map(trajectory, costmap.shape[0])
        # _, _, p = plan_paths(1, costmap, workspace, starts=[starts[i]],
        #                     targets=[targets[i]])
        loss_augmentation = scaled_hamming_loss_map(trajectory, costmap.shape[0],
                                                    loss_scalar, loss_stddev)

        # Push up on loss augmented state frequency
        try:
            esf += get_expected_edge_frequency(costmap - loss_augmentation, N,
                                               costmap.shape[0], starts, targets,
                                               workspace)
        except Exception as e:
            print("Exception happened while computing expected state frequencies")
            print(e)
            raise
    # Scaleing
    esf = esf / esf.sum() * - map.sum()
    map = esf + map
    return map


def get_avg_learch_esf_target():
    # TODO implement
    print("please implement")
