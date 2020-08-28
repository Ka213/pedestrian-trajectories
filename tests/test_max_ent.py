import common_import
from pyrieef.geometry.workspace import *
from my_utils.output_costmap import *
from my_learning.max_ent import *


def test_maxEnt():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 200

    workspace = Workspace()
    np.random.seed(1)
    # Create random costmap
    w, original_costmap, starts, targets, paths = \
        create_random_environment(nb_points, nb_rbfs, sigma, nb_samples,
                                  workspace)
    centers = workspace.box.meshgrid_points(nb_rbfs)

    # Learn costmap
    a = MaxEnt(nb_points, centers, sigma, paths, starts, targets, workspace)
    maps, w_t = a.solve()
    print("weight difference: ", (np.absolute(w_t - w)).sum())
    assert (np.absolute(w_t - w)).sum() < nb_points ** 2 * 0.4
    # Output learned costmaps
    show_multiple(maps, [original_costmap], workspace, show_result,
                  directory=home + '/../figures/maxEnt.png')


if __name__ == "__main__":
    show_result = 'SHOW'
    test_maxEnt()
