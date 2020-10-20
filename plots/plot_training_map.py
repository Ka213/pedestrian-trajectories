import common_import

from my_utils.environment import *
from my_utils.output_costmap import *
from my_utils.my_utils import *
from nn.nn_utils import *

show_result = 'SHOW'
algorithm = 'loss_aug_occ'
nb_points = 28
nb_rbfs = 4
sigma = 0.1
nb_samples = 1

loss_stddev = 3
loss_scalar = 1
N = 35

workspace = Workspace()
np.random.seed(1)

w, original_costmap, starts, targets, paths, centers = \
    create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples,
                            Workspace())
phi = get_phi(nb_points, centers, sigma, Workspace())

map = np.zeros((nb_points, nb_points))
if algorithm == 'learch':
    map = get_costmap(phi, np.exp(np.ones((nb_rbfs ** 2))))
    map, op = get_learch_input(map, paths, starts, targets, loss_stddev,
                               loss_scalar)
elif algorithm == 'maxEnt':
    map = get_maxEnt_input(map, N, paths, targets, phi)
elif algorithm == 'occ':
    map = get_occ_input(map, N, paths, targets)
elif algorithm == 'loss_aug_occ':
    map = get_loss_aug_occ_input(map, paths, targets, loss_scalar,
                                 loss_stddev, N)
elif algorithm == 'occ_learch':
    map = get_avg_learch_occ_input()

# Output costmap
# startmap = get_costmap(phi, np.exp(np.ones(nb_rbfs ** 2)))
# _,_, op = plan_paths(nb_samples, startmap, workspace, starts, targets)


show_multiple([map], [original_costmap], workspace, show_result
              # ,starts=starts, targets=targets, paths=paths, optimal_paths=[op]
              )
show(map, workspace, show_result
     # ,starts=starts, targets=targets, paths=paths, optimal_paths=[op]
     )
