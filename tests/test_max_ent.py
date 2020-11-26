import common_import

from my_learning.max_ent import *
from my_utils.output_costmap import *
from pyrieef.geometry.workspace import *


def test_maxEnt():
    nb_points = 28
    nb_rbfs = 4
    sigma = 0.15
    nb_samples = 50
    nb_env = 20

    workspace = Workspace()
    m = MaxEnt(nb_points, nb_rbfs, sigma, workspace)
    costs = []
    starts_gt = []
    targets_gt = []
    demonstrations = []
    for i in range(nb_env):
        np.random.seed(i)
        # Create random costmap
        w, costmap_gt, starts, targets, paths, centers = \
            create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples,
                                    workspace)
        costs.append(costmap_gt)
        demonstrations.append(paths)
        starts_gt.append(starts)
        targets_gt.append(targets)
        # Learn costmap
        m.add_environment(centers, paths, starts, targets)

    maps, ex_paths, w_t, step = m.solve()

    print("weight difference: ", (np.absolute(w_t - w)).sum())
    assert (np.absolute(w_t - w)).sum() < nb_rbfs ** 2 * 0.3
    # Output learned costmaps
    show_multiple(maps, costs, workspace, show_result,
                  directory=home + '/../figures/maxEnt.png')


if __name__ == "__main__":
    show_result = 'SHOW'
    test_maxEnt()
