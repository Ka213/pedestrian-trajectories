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
    nb_samples = 100

    workspace = Workspace()

    w, original_costmap, starts, targets, paths = \
        create_random_environment(nb_points, nb_rbfs, sigma, nb_samples,
                                  workspace)

    centers = workspace.box.meshgrid_points(nb_rbfs)

    # Learn costmap
    l = Learch2D(nb_points, centers, sigma, paths, starts, targets, workspace)
    # Set hyperparemeters
    l._loss_scalar = x["loss scalar"]
    l._loss_stddev = x["loss stddev"]
    l._learning_rate = x["learning rate"]
    l._stepsize_scalar = x["step size scalar"]
    l._l2_regularizer = x["l2 regularizer"]
    l._proximal_regularizer = x["proximal regularizer"]
    l.exponentiated_gd = True
    l.initialize_mydata()

    maps, optimal_paths, w = l.solve()
    # Calculate Training loss
    loss = get_learch_loss(original_costmap, optimal_paths[-1], paths,
                           nb_samples, l._l2_regularizer,
                           l._proximal_regularizer, w)
    if loss < 0:
        loss = sys.maxsize
    return loss


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()
loss_scalar = UniformIntegerHyperparameter("loss scalar", 0, 20,
                                           default_value=14)
loss_stddev = UniformIntegerHyperparameter("loss stddev", 0, 20,
                                           default_value=7)
learning_rate = UniformFloatHyperparameter("learning rate", 0.1, 2.0,
                                           default_value=1)
step_size_scalar = UniformIntegerHyperparameter("step size scalar", 1, 20,
                                                default_value=16)
l2_regularizer = UniformFloatHyperparameter("l2 regularizer", 0.001, 1,
                                            default_value=1, log='True')
proximal_regularizer = UniformFloatHyperparameter("proximal regularizer", 0.001,
                                                  1, default_value=0.001,
                                                  log='True')
cs.add_hyperparameters([loss_scalar, loss_stddev, learning_rate,
                        step_size_scalar, l2_regularizer, proximal_regularizer])

# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality
                     # (alternatively runtime)
                     "runcount-limit": 2,
                     "cs": cs,  # configuration space
                     "deterministic": "false",
                     "shared_model": True,
                     "input_psmac_dirs": "smac-output-maxEnt",
                     "cutoff_time": 1000,
                     "wallclock_limit": 10000
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

# inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
# print("Optimized Value: %.2f" % inc_value)
