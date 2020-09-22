import common_import
from my_utils.output_costmap import *
from my_learning.learch import *
from pyrieef.geometry.workspace import Workspace


def test_supervised_learning():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1

    workspace = Workspace()

    l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
    np.random.seed(0)
    # Create random costmap
    w, original_costmap, starts, targets, paths, centers = \
        load_environment("environment_sample_centers0")  # \
    #    create_random_environment(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    # Learn costmap
    l.add_environment(centers, paths, starts, targets)

    # adjust parameters
    l._learning_rate = 1
    l._stepsize_scalar = 1
    l._l2_regularizer = 1e-10
    l._proximal_regularizer = 0

    # iterate LEARCH until weights are converged
    w_old = copy.deepcopy(l.w)
    e = 10
    i = 0

    while e > 1e-10:
        print("step: ", i)
        l.instances[0].update(l.w)
        # construct D from all states of the original analysis
        x, y = np.meshgrid(range(nb_points), range(nb_points))
        x_1 = np.reshape(y, nb_points ** 2)
        x_2 = np.reshape(x, nb_points ** 2)
        y = np.reshape(original_costmap -
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
        # show_multiple([l.instances[0].costmap], [original_costmap], workspace,
        #              show_result, weights=[w_t], centers=centers)

    costmap = np.tensordot(np.log(l.w), l.instances[0].phi, axes=1)
    costs = np.sum(np.abs(costmap - original_costmap))
    print("costs: ", costs)
    assert np.average(costs) < 1e-4
    show_multiple(l.instances[0].learned_maps[::200], [original_costmap],
                  workspace, show_result, directory=home +
                                                    '/../results/figures/test_supervised_learning.png')

def test_D():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 5

    workspace = Workspace()

    l = Learch2D(nb_points, nb_rbfs, sigma, workspace)

    np.random.seed(0)
    # Create random costmap
    w, original_costmap, starts, targets, paths, centers = \
        load_environment("environment_sample_centers0")  # \
    #    create_random_environment(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    starts = starts[:nb_samples]
    targets = targets[:nb_samples]
    paths = paths[:nb_samples]
    # Learn costmap
    l.add_environment(centers, paths, starts, targets)

    D, _ = l.instances[0].planning()
    len_paths = [len(p) for p in paths]
    len_optimal_paths = [len(op) for op in l.instances[0].optimal_paths[-1]]
    assert D.shape == (3, np.sum(len_paths) + np.sum(len_optimal_paths))
    print(D)
    show(l.instances[0].costmap, workspace, show_result, starts=starts,
         targets=targets, paths=paths,
         optimal_paths=l.instances[0].optimal_paths[-1], d=D,
         directory=home + '/../results/figures/map_with_D.png')




if __name__ == "__main__":
    show_result = 'SHOW'
    test_supervised_learning()
    test_D()
