import common_import
import logging

import numpy as np
from ConfigSpace.hyperparameters import *
from my_learning.max_ent import *
from my_utils.output_analysis import *

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario


# Run maxEnt
def maxEnt(x):
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 60
    nb_env = 1

    workspace = Workspace()
    m = MaxEnt(nb_points, nb_rbfs, sigma, workspace)
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
        original_paths.append(paths)
        original_starts.append(starts)
        original_targets.append(targets)
        # Learn costmap
        m.add_environment(centers, paths, starts, targets)

    m._learning_rate = x["learning rate"]
    m._stepsize_scalar = x["step size scalar"]
    for _, i in enumerate(m.instances):
        i._N = x["N"]

    maps, optimal_paths, w_t, step = m.solve()

    loss = np.average(get_maxEnt_loss(maps, original_paths, nb_samples * nb_env))

    return loss


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()
N = UniformIntegerHyperparameter("N", 50, 200, default_value=100)
learning_rate = UniformFloatHyperparameter("learning rate", 0.1, 2.0,
                                           default_value=1)
step_size_scalar = UniformIntegerHyperparameter("step size scalar", 1, 20,
                                                default_value=1)
cs.add_hyperparameters([N, learning_rate, step_size_scalar])

# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality
                     # (alternatively runtime)
                     "runcount-limit": 10,
                     "cs": cs,  # configuration space
                     "deterministic": "false",
                     "shared_model": True,
                     "input_psmac_dirs": "smac-output-maxEnt",
                     "cutoff_time": 9000,
                     "wallclock_limit": 'inf'
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = maxEnt(cs.get_default_configuration())
print("Default Value: %.2f" % def_value)

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC4HPO(scenario=scenario,
                rng=np.random.RandomState(42),
                tae_runner=maxEnt)

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

