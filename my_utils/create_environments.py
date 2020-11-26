import common_import

from my_utils.environment import *
from my_utils.output_costmap import *

show_result = 'SHOW'
nb_points = 28
nb_rbfs = 4
sigma = 0.15
nb_samples = 500
nb_environments = 20

workspace = Workspace()

# Create random environments
for i in range(nb_environments):
    print(i)
    np.random.seed(i)
    w, costmap, starts, targets, paths, centers = \
        create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    save_environment("environment_rbfs_28_" + str(i), nb_points, nb_rbfs,
                     sigma, centers, nb_samples, w, costmap, starts, targets,
                     np.asarray(paths))
