import common_import

from my_utils.my_utils import *
import tensorflow as tf
from pyrieef.learning.tf_networks import *
from nn.random_environment_sdf import *
from nn.dataset_sdf import *
import matplotlib.pyplot as plt
from matplotlib import cm


class One_Vector_Sdf():
    """
    """

    def __init__(self, data, learning, workspace):
        self.workspace = workspace
        self.learning = learning

        self.PIXELS = 28  # Used to be 100.
        self.NUM_TEST = 0
        self.NUM_TRAIN = None

        self.costmaps = None
        self.datafile = data
        self.workspaces = None
        self.costs = None

        self.decoded_data_train = []
        self.decoded_data_test = []
        self.learned_maps = []

        self.initialize()

    def initialize(self):
        tf.set_random_seed(1)
        # Costmaps
        self.costmaps = get_dataset_id(data_id=self.datafile)
        self.costmaps.normalize_maps()
        self.costmaps.reshape_data_to_tensors()

        assert self.NUM_TEST <= len(self.costmaps.test_inputs)
        self.NUM_TRAIN = self.costmaps.num_examples

        self.workspaces = load_workspace_dataset(basename=self.datafile
                                                          + '.hdf5')

        self.costs = np.ones((self.costmaps._max_index, self.PIXELS,
                              self.PIXELS))
        self.costs = np.exp(np.array(self.costs))

    def n_steps(self, n, begin=0):
        """ Do n steps of the algorithm over multiple environments """
        for i in range(n):
            self.learned_maps.append(np.log(self.costs))
        return np.log(self.costs), None, n

    def solve(self, begin=0):
        """ Compute the algorithm for multiple environments
            until the weights converge
        """
        return np.log(self.costs), None, 0
