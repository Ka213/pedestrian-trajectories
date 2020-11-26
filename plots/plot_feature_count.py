import common_import

from my_utils.output_costmap import *
from my_utils.environment import *
from my_utils.my_utils import *

show_result = 'SHOW'
nb_points = 28
nb_rbfs = 4
sigma = 0.15
nb_samples = 1

workspace = Workspace()
np.random.seed(1)
# Create random costmap
w, costs, starts, targets, paths, centers = \
    create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)
phi = get_phi(nb_points, centers, sigma, workspace)
# Calculate feature counts
f = get_empirical_feature_count(paths, phi)

map1 = get_costmap(phi, f)
f = - f - np.min(-f)
map2 = get_costmap(phi, f)

show(costs, workspace, show_result, starts=starts, targets=targets, paths=paths,
     centers=centers, weights=np.ones(nb_rbfs ** 2),
     directory=home + '/../results/figures/feature_count_gt.pdf')
show(map1, workspace, show_result, starts=starts, targets=targets, paths=paths,
     centers=centers, weights=np.ones(nb_rbfs ** 2),
     directory=home + '/../results/figures/feature_count.pdf')
show_3D(map1, workspace, show_result, starts=starts, targets=targets,
        paths=paths, centers=centers,
        directory=home + '/../results/figures/feature_count3D.pdf')
show_multiple_3D([map1, map2, costs], workspace, show_result,
                 labels=['feature count', 'inverse', 'original'],
                 starts=starts, targets=targets, paths=paths)
show_multiple([map1, map2], [costs], workspace, show_result,
              starts=starts, targets=targets, paths=paths,
              directory=home + '/../results/figures/feature_count_multi.png',
              title="empirical feature count")
