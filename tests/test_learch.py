import common_import

from my_learning.learch import *
from my_utils.output_costmap import *
from pyrieef.geometry.workspace import Workspace


def test_supervised_learning():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 20

    workspace = Workspace()

    l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
    np.random.seed(0)
    # Create random costmap
    w, costs, starts, targets, paths, centers = \
        create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    # Learn costmap
    l.add_environment(centers, paths, starts, targets)

    # Adjust parameters
    l._learning_rate = 1
    l._stepsize_scalar = 1
    l._l2_regularizer = 1e-10
    l._proximal_regularizer = 0

    # Iterate LEARCH until weights are converged
    w_old = copy.deepcopy(l.w)
    e = 10
    i = 0

    while e > 1e-3:
        print("step: ", i)
        l.instances[0].update(l.w)
        # construct D from all states of the original analysis
        x, y = np.meshgrid(range(nb_points), range(nb_points))
        x_1 = np.reshape(y, nb_points ** 2)
        x_2 = np.reshape(x, nb_points ** 2)
        y = np.reshape(costs -
                       l.instances[0].learned_maps[-1], nb_points ** 2)
        D = np.vstack((x_1, x_2, y))
        w_t = l.instances[0].supervised_learning(D)
        l.w = l.w * np.exp(get_stepsize(i, l._learning_rate,
                                        l._stepsize_scalar) * w_t)
        print("step size:", get_stepsize(i, l._learning_rate,
                                         l._stepsize_scalar))
        e = np.amax(np.absolute(l.w - w_old))
        print("convergence: ", e)
        w_old = copy.deepcopy(l.w)
        i += 1
        # show_multiple([l.instances[0].costmap], [costs], workspace,
        #              show_result, weights=[w_t], centers=centers)

    costmap = np.tensordot(np.log(l.w), l.instances[0].phi, axes=1)
    _, _, p = plan_paths(nb_samples, costmap, workspace,
                         starts=starts, targets=targets)
    c = get_learch_loss([costs], [p], [paths], nb_samples)
    print("costs: ", c)
    assert np.average(c) < 1e-10
    show_multiple(l.instances[0].learned_maps, [costs], workspace,
                  show_result, directory=
                  home + '/../results/figures/test_supervised_learning.png')

def test_D():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 5

    workspace = Workspace()

    l = Learch2D(nb_points, nb_rbfs, sigma, workspace)

    np.random.seed(0)
    # Create random costmap
    w, costs, starts, targets, paths, centers = \
        create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    starts = starts[:nb_samples]
    targets = targets[:nb_samples]
    paths = paths[:nb_samples]
    # Learn costmap
    l.add_environment(centers, paths, starts, targets)

    D, ex = l.instances[0].planning()
    len_paths = [len(p) for p in paths]
    len_ex_paths = [len(p) for p in ex]
    assert D.shape == (3, np.sum(len_paths) + np.sum(len_ex_paths))
    print(D)
    show(l.instances[0].costmap, workspace, show_result, starts=starts,
         targets=targets, paths=paths,
         ex_paths=ex, d=D,
         directory=home + '/../results/figures/map_with_D.png')


def test_linear_regression_with_zero_states():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 20
    nb_env = 5

    workspace = Workspace()

    l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
    costs = []
    starts_gt = []
    targets_gt = []
    demonstrations = []
    for i in range(nb_env):
        np.random.seed(i)
        # Create random costmap
        w, costmap_gt, starts, targets, paths, centers = \
            create_rand_env(nb_points, nb_rbfs, sigma, nb_samples, workspace)
        costs.append(costmap_gt)
        starts = starts[:nb_samples]
        targets = targets[:nb_samples]
        paths = paths[:nb_samples]
        demonstrations.append(paths)
        starts_gt.append(starts)
        targets_gt.append(targets)
        # Learn costmap
        l.add_environment(centers, paths, starts, targets)

    for j, i in enumerate(l.instances):
        D, ex_paths = i.planning()
        for k in range(nb_points):
            for n in range(nb_points):
                x = np.where(D[0] == k)
                y = np.where(D[1] == n)
                if len(np.intersect1d(x[0], y[0])) == 0:
                    d = np.vstack((k, n, 0))
                    D = np.hstack((D, d))
        w_new = i.supervised_learning(D)

        D, ex_paths = i.planning()
        w = i.supervised_learning(D)

        print("new weights with other states put to zero: ", w_new)
        print("new weights without other states put to zero: ", w)
        assert not np.allclose(w, w_new, atol=1e-13, rtol=1e-14)
        assert not np.array_equal(w, w_new)


if __name__ == "__main__":
    show_result = 'SHOW'
    test_supervised_learning()
    test_D()
    test_linear_regression_with_zero_states()
