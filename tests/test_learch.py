import common_import
from my_utils.output_costmap import *
from my_learning.learch import *
from pyrieef.geometry.workspace import Workspace


def test_supervised_learning():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 10

    workspace = Workspace()
    np.random.seed(1)
    # Create random costmap
    w, original_costmap, starts, targets, paths = \
        create_random_environment(nb_points, nb_rbfs, sigma, nb_samples,
                                  workspace)
    centers = workspace.box.meshgrid_points(nb_rbfs)

    l = Learch2D(nb_points, centers, sigma, paths, starts, targets, workspace)
    l.exponentiated_gd = True
    if l.exponentiated_gd:
        t = 1
    else:
        t = 0.001
    l.D = np.zeros((3, nb_points ** 2))

    # adjust parameters
    l._learning_rate = 1
    l._stepsize_scalar = 1
    l._l2_regularizer = 1
    l._proximal_regularizer = 0

    # iterate LEARCH until weights are converged
    w_old = copy.deepcopy(l.w)
    e = 10
    i = 0

    while e > t:
        # construct D from all states of the original analysis
        x, y = np.meshgrid(range(nb_points), range(nb_points))
        x_1 = np.reshape(y, nb_points ** 2)
        x_2 = np.reshape(x, nb_points ** 2)
        y = np.reshape(original_costmap, nb_points ** 2)
        l.D = np.vstack((x_1, x_2, y))
        l.supervised_learning(i)
        maps = l.maps
        e = np.amax(np.absolute(l.weights[-1] - w_old))
        print("convergence: ", e)
        w_old = copy.deepcopy(l.weights[-1])
        i += 1
    print((np.absolute(maps[-1] - original_costmap)).sum())
    assert (np.absolute(maps[-1] - original_costmap)).sum() < \
           nb_points ** 2 * 1.5
    show_multiple(maps[::40], original_costmap, workspace, show_result, directory=
    home + '/../figures/maps_from_test_supervised_learning.png')

def test_D():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 5

    workspace = Workspace()
    np.random.seed(1)
    # Create random costmap
    w, original_costmap, starts, targets, paths = \
        create_random_environment(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    centers = workspace.box.meshgrid_points(nb_rbfs)

    l = Learch2D(nb_points, centers, sigma, paths, starts, targets, workspace)
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
