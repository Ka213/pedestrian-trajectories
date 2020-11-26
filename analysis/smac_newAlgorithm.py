import common_import
import logging

import numpy as np
from ConfigSpace.hyperparameters import *
from my_learning.learch_avg_esf_path import *
from my_utils.output_analysis import *

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario


# Run LEARCH variant
def learch_variant(x):
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 100
    nb_env = 1

    workspace = Workspace()

    l = Learch_Avg_Esf_Path(nb_points, nb_rbfs, sigma, workspace)
    costs = []
    starts_gt = []
    targets_gt = []
    demonstrations = []
    for i in range(nb_env):
        np.random.seed(i)
        # Create random costmap
        w, original_costmap, starts, targets, paths, centers = \
            create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace)
        costs.append(original_costmap)
        demonstrations.append(paths)
        starts_gt.append(starts)
        targets_gt.append(targets)
        # Learn costmap
        l.add_environment(centers, paths, starts, targets)

    l._learning_rate = x["learning rate"]
    l._stepsize_scalar = x["step size scalar"]
    for _, i in enumerate(l.instances):
        i._l2_regularizer = x["l2 regularizer"]
        # i._proximal_regularizer = x["proximal regularizer"]
        i._N = x["N"]

    try:
        maps, ex_paths, w_t, step = l.solve()

    except Warning as w:
        return sys.maxsize

    loss = np.average(get_edt_loss(nb_points, ex_paths, demonstrations,
                                   nb_samples))
    return loss


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()
learning_rate = UniformIntegerHyperparameter("learning rate", 1, 20,
                                             default_value=10)
step_size_scalar = UniformIntegerHyperparameter("step size scalar", 1, 20,
                                                default_value=1)
l2_regularizer = UniformIntegerHyperparameter("l2 regularizer", 1, 1000,
                                              default_value=10, log='True')
proximal_regularizer = UniformIntegerHyperparameter("proximal regularizer", 1,
                                                    100, default_value=1,
                                                    log='True')
N = UniformIntegerHyperparameter("N", 50, 200, default_value=100)
cs.add_hyperparameters([learning_rate, step_size_scalar, l2_regularizer, N])

# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality
                     # (alternatively runtime)
                     "runcount-limit": 2,
                     "cs": cs,  # configuration space
                     "deterministic": "false",
                     "shared_model": True,
                     "input_psmac_dirs": "smac-output-learch",
                     "cutoff_time": 9000,
                     "wallclock_limit": 'inf'
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = learch_variant(cs.get_default_configuration())
print("Default Value: %.2f" % def_value)

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC4HPO(scenario=scenario,
                rng=np.random.RandomState(42),
                tae_runner=learch_variant)

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent
