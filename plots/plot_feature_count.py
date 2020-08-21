import common_import

from my_utils.output_costmap import *
from my_utils.environment import *
from my_utils.my_utils import *

show_result = 'SHOW'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 100

workspace = Workspace()
np.random.seed(1)
# Create random costmap
w, original_costmap, starts, targets, paths = \
    create_random_environment(nb_points, nb_rbfs, sigma, nb_samples, workspace)
centers = workspace.box.meshgrid_points(nb_rbfs)
Phi = get_phi(nb_points, centers, sigma, workspace)
# Calculate feature counts
f = get_empirical_feature_count(paths, Phi)
f = - f - np.min(-f)

map = get_costmap(nb_points, centers, sigma, f, workspace)
show_multiple([map], original_costmap, workspace, show_result,
              # starts=starts, targets=targets, paths=paths,
              directory=home + '/../figures/feature_count.png',
              title="empirical feature count")
