import common_import

import numpy as np
from my_utils.output import *
from costmap.costmap import *
from learch.learch import *
from pyrieef.geometry.workspace import Workspace


def test_supervised_learning():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 10

    workspace = Workspace()
    np.random.seed(3)

    # Create costmap with rbfs
    w = np.random.random(nb_rbfs ** 2)
    centers = workspace.box.meshgrid_points(nb_rbfs)
    original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)

    # Plan example trajectories
    starts, targets, paths = plan_paths(nb_samples, original_costmap, workspace)

    l = Learch2D(nb_points, centers, sigma, paths, starts,
                 targets, workspace)
    l.D = np.zeros((3, nb_points ** 2))

    # adjust parameters
    l._learning_rate = 1
    l._stepsize_scalar = 1
    l._l2_regularizer = 0
    l._proximal_regularizer = 0

    # iterate LEARCH until weights are converged
    w_old = copy.deepcopy(l.w)
    e = 10
    i = 0
    while e > 0.001:
        # construct D from all states of the original costmap
        x, y = np.meshgrid(range(nb_points), range(nb_points))
        x_1 = np.reshape(y, nb_points ** 2)
        x_2 = np.reshape(x, nb_points ** 2)
        y = np.reshape(original_costmap, nb_points ** 2)
        l.D = np.vstack((x_1, x_2, y))
        maps, _, w = l.one_step(i)
        e = (np.absolute(w[-1] - w_old)).sum()
        print("convergence: ", e)
        w_old = copy.deepcopy(w[-1])
        i += 1
    assert (np.absolute(maps[-1] - original_costmap)).sum() < \
           nb_points ** 2 * 1.5
    show_multiple(maps, original_costmap, workspace, show_result, directory=
    home + '/../figures/maps_from_test_supervised_learning')


def test_D():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 5

    workspace = Workspace()
    np.random.seed(1)

    # Create costmap with rbfs
    w = np.random.random(nb_rbfs ** 2)
    centers = workspace.box.meshgrid_points(nb_rbfs)
    original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)

    # Plan example trajectories
    starts, targets, paths = plan_paths(nb_samples, original_costmap, workspace)

    l = Learch2D(nb_points, centers, sigma, paths, starts,
                 targets, workspace)
    l.planning()
    len_paths = [len(p) for p in paths]
    len_optimal_paths = [len(op) for op in l.optimal_paths[-1]]
    assert l.D.shape == (3, np.sum(len_paths) + np.sum(len_optimal_paths))
    print(l.D)
    show(l.costmap, workspace, show_result, starts=starts, targets=targets,
         paths=paths, optimal_paths=l.optimal_paths[-1], d=l.D,
         directory=home + '/../figures/map_with_D')


if __name__ == "__main__":
    show_result = 'SHOW'
    test_supervised_learning()
    test_D()
