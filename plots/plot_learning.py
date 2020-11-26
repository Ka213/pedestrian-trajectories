import common_import

from my_learning.learch_avg_esf_path import *
from my_learning.learch_esf import *
from my_learning.learch_loss_aug_esf import *
from my_learning.random import *
from my_learning.learch import *
from my_learning.max_ent import *
from my_utils.environment import *
from my_utils.my_utils import *
from my_utils.output_costmap import *

show_result = 'SHOW'
# set the learning method to evaluate
# choose between learch, maxEnt, avg_esf_path, esf, loss_aug_esf
# oneVector and random
learning = 'random'
nb_points = 28
nb_rbfs = 4
sigma = 0.15
nb_samples = 20
nb_env = 3
nb_training = 20

workspace = Workspace()
np.random.seed(0)

# Learn costmap
if learning == 'learch':
    l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'maxEnt':
    l = MaxEnt(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'avg_esf_path':
    l = Learch_Avg_Esf_Path(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'esf':
    l = Learch_Esf(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'loss_aug_esf':
    l = Learch_Loss_Aug_Esf(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'oneVector':
    l = Learning(nb_points, nb_rbfs, sigma, workspace)
elif learning == 'random':
    l = Random(nb_points, nb_rbfs, sigma, workspace)

# Get training samples
costs = []
starts_gt = []
targets_gt = []
demonstrations = []
for i in range(nb_env):
    np.random.seed(i)
    # Create random costmap
    w, costmap_gt, s, t, p, centers = \
        create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)
    costs.append(costmap_gt)
    starts = s[:nb_samples]
    targets = t[:nb_samples]
    paths = p[:nb_samples]

    demonstrations.append(p[:nb_training])
    starts_gt.append(s[:nb_training])
    targets_gt.append(t[:nb_training])
    # Learn costmap
    l.add_environment(centers, paths, starts, targets)

learned_maps, ex_paths, w_t, step_count = l.solve()


# Calculate training loss
training_loss_l = get_learch_loss(costs, ex_paths, demonstrations, nb_training)
print("learch loss: ", np.average(training_loss_l))
training_loss_m = get_maxEnt_loss(learned_maps, demonstrations, nb_training)
print("maxEnt loss: ", np.average(training_loss_m))
training_edt = get_edt_loss(nb_points, ex_paths, demonstrations, nb_training)
print("edt loss: ", np.average(training_edt))
training_costs = get_overall_loss(learned_maps, costs)
print("cost loss: ", np.average(training_costs))
training_nll = get_nll(ex_paths, demonstrations, nb_points, nb_training)
print("nll loss: ", np.average(training_nll))

for j, i in enumerate(l.instances):
    show_predictions(learned_maps[j], costs[j], workspace, show_result,
                     starts=starts_gt[j], targets=targets_gt[j],
                     paths=demonstrations[j], ex_paths=ex_paths[j],
                     directory=home + '/../results/figures/{}_{}results'.
                     format(learning, nb_samples))
