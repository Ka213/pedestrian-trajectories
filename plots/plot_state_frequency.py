import common_import

from my_utils.output import *
from my_utils.costmap import *
from my_utils.my_utils import *

show_result = 'SHOW'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 1
N = 40

workspace = Workspace()
np.random.seed(1)

# Create costmap with rbfs
w = np.random.random(nb_rbfs ** 2)
centers = workspace.box.meshgrid_points(nb_rbfs)
original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)
Phi = get_phi(nb_points, centers, sigma, workspace)
# Plan example trajectories
starts, targets, paths = plan_paths(nb_samples, original_costmap, workspace)
P = get_transition_probabilities(original_costmap, nb_points)

D = get_expected_edge_frequency(P, original_costmap, N, nb_points,
                                targets, paths, workspace)

show(D, workspace, show_result, starts=starts, targets=targets, paths=paths,
     directory=home + '/../figures/stateFrequency.png',
     title="expected state frequeny")

D = - D - np.min(-D)
f = np.tensordot(Phi, D)
map = get_costmap(nb_points, centers, sigma, f, workspace)
show_multiple([map], original_costmap, workspace, show_result,
              # starts=starts, targets=targets, paths=paths,
              directory=home + '/../figures/stateFrequencyCostmap.png',
              title='expected state frequency map')
