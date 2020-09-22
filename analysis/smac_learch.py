import common_import
import logging

import numpy as np
from ConfigSpace.hyperparameters import *
from my_learning.learch import *
from my_utils.output_analysis import *

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario


# Run LEARCH
def learch(x):
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 60
    nb_env = 1

    workspace = Workspace()

    l = Learch2D(nb_points, nb_rbfs, sigma, workspace)
    original_costmaps = []
    original_starts = []
    original_targets = []
    original_paths = []
    for i in range(nb_env):
        # np.random.seed(i)
        # Create random costmap
        w, original_costmap, starts, targets, paths, centers = \
            create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples,
                                    workspace)
        original_costmaps.append(original_costmap)
        starts = starts[:nb_samples]
        targets = targets[:nb_samples]
        paths = paths[:nb_samples]
        original_paths.append(paths)
        original_starts.append(starts)
        original_targets.append(targets)
        # Learn costmap
        l.add_environment(centers, paths, starts, targets)

    l._learning_rate = x["learning rate"]
    l._stepsize_scalar = x["step size scalar"]
    for _, i in enumerate(l.instances):
        i._loss_scalar = x["loss scalr"]
        i._loss_stddev = x["loss stddev"]
        i._l2_regularizer = x["l2 regularizer"]
        i._proximal_regularizer = x["proximal regularizer"]

    maps, optimal_paths, w_t, step = l.solve()

    loss = - np.average(get_learch_loss(original_costmaps, optimal_paths,
                                        original_paths, nb_samples * nb_env))
    if loss < 0:
        loss = sys.maxsize
    return loss


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()
loss_scalar = UniformIntegerHyperparameter("loss scalar", 0, 20,
                                           default_value=1)
loss_stddev = UniformIntegerHyperparameter("loss stddev", 0, 20,
                                           default_value=10)
learning_rate = UniformFloatHyperparameter("learning rate", 0.1, 2.0,
                                           default_value=1)
step_size_scalar = UniformIntegerHyperparameter("step size scalar", 1, 20,
                                                default_value=1)
l2_regularizer = UniformFloatHyperparameter("l2 regularizer", 0.001, 1,
                                            default_value=0.01, log='True')
proximal_regularizer = UniformFloatHyperparameter("proximal regularizer", 0.001,
                                                  1, default_value=0.001,
                                                  log='True')
cs.add_hyperparameters([loss_scalar, loss_stddev, learning_rate,
                        step_size_scalar, l2_regularizer, proximal_regularizer])

# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality
                     # (alternatively runtime)
                     "runcount-limit": 50,
                     "cs": cs,  # configuration space
                     "deterministic": "false",
                     "shared_model": True,
                     "input_psmac_dirs": "smac-output-learch",
                     "cutoff_time": 9000,
                     "wallclock_limit": 'inf'
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = learch(cs.get_default_configuration())
print("Default Value: %.2f" % def_value)

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC4HPO(scenario=scenario,
                rng=np.random.RandomState(42),
                tae_runner=learch)

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

