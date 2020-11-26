import common_import

from my_utils.output_costmap import *
from my_utils.environment import *
from my_utils.my_utils import *

show_result = 'SHOW'
loss_augmentation = False
nb_points = 28
nb_rbfs = 4
sigma = 0.15
nb_samples = 3
N = 45

workspace = Workspace()
np.random.seed(1)
# Create random costmap
w, costs, starts, targets, paths, centers = \
    create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)

phi = get_phi(nb_points, centers, sigma, workspace)

# Calculate state frequency
# compute expected state frequency of ground truth costmap
d = get_expected_edge_frequency(costs, N, nb_points, starts, targets,
                                workspace)

f_expected = np.tensordot(phi, d)
costmap1 = get_costmap(phi, - f_expected - np.min(- f_expected))
costmap2 = get_costmap(phi, f_expected)

show_3D(d, workspace, show_result, starts=starts, targets=targets, paths=paths,
        centers=centers)
show_3D(costmap2, workspace, show_result, starts=starts, targets=targets,
        paths=paths, centers=centers)
show(costs, workspace, show_result, starts=starts, targets=targets, paths=paths,
     directory=home + '/../results/figures/expectedStateFrequency_gt.pdf')
show(d, workspace, show_result, starts=starts, targets=targets, paths=paths,
     directory=home + '/../results/figures/expectedStateFrequency.pdf')
show(costmap2, workspace, show_result, starts=starts, targets=targets,
     paths=paths, centers=centers, weights=np.ones(nb_rbfs ** 2),
     directory=home + '/../results/figures/stateFrequencyCostmap.pdf')
show_multiple([d, costmap1, costmap2], [costs], workspace, show_result,
              # starts=starts, targets=targets, paths=paths,
              directory=home + '/../results/figures/stateFrequency_multi.pdf')  # ,
# title='expected state frequency map')
