import common_import
from pyrieef.geometry.workspace import *
from my_utils.output_costmap import *
from my_learning.max_ent import *


def test_maxEnt():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 500
    nb_env = 1

    workspace = Workspace()
    m = MaxEnt(nb_points, nb_rbfs, sigma, workspace)
    original_costmaps = []
    original_starts = []
    original_targets = []
    original_paths = []
    for i in range(nb_env):
        np.random.seed(i)
        # Create random costmap
        w, original_costmap, starts, targets, paths, centers = \
            load_environment("environment_sample_centers" + str(i))  # \
        #    create_random_environment(nb_points, nb_rbfs, sigma, nb_samples, workspace)
        original_costmaps.append(original_costmap)
        starts = starts  # [:nb_samples]
        targets = targets  # [:nb_samples]
        paths = paths  # [:nb_samples]
        original_paths.append(paths)
        original_starts.append(starts)
        original_targets.append(targets)
        # Learn costmap
        m.add_environment(centers, paths, starts, targets)

    maps, optimal_paths, w_t, step = m.solve()

    print("weight difference: ", (np.absolute(w_t - w)).sum())
    assert (np.absolute(w_t - w)).sum() < nb_rbfs ** 2 * 0.6
    # Output learned costmaps
    show_multiple(maps, original_costmaps, workspace, show_result,
                  directory=home + '/../figures/maxEnt.png')


if __name__ == "__main__":
    show_result = 'SHOW'
    test_maxEnt()
