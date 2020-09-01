import common_import

from environment import *
from output_costmap import *
from my_utils import *

show_result = 'SHOW'
with_trajectories = True
average_cost = False
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 500
nb_environments = 5

workspace = Workspace()

# Create random environments
for i in range(nb_environments):
    print(i)
    np.random.seed(i)
    w, costmap, starts, targets, paths, centers = \
        create_rand_env(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    save_environment("environment" + str(i), nb_points, nb_rbfs,
                     sigma, centers, nb_samples, w, costmap, starts, targets,
                     np.asarray(paths))
