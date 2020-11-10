import common_import

from my_utils.my_utils import *
import tensorflow as tf
from pyrieef.learning.tf_networks import *
from nn.random_environment_sdf import *
from nn.dataset_sdf import *
import matplotlib.pyplot as plt
from matplotlib import cm


class Deep_Learning():
    """
    """

    def __init__(self, data, learning, workspace):
        self.DRAW = True
        self.SAVE_TO_FILE = False

        self.workspace = workspace
        self.learning = learning

        # Parameters to compute the step size
        self._learning_rate = 1
        self._stepsize_scalar = 1

        self.convergence = 0.1

        self._loss_scalar = 1
        self._loss_stddev = 10
        self._N = 35

        self.BATCHES = 8000
        self.BATCH_SIZE = 64
        self.PIXELS = 28  # Used to be 100.
        self.LR = 0.002  # learning rate
        self.NUM_TEST = 0
        self.NUM_TRAIN = None

        self.network = ConvDeconvResize()
        self.costmaps = None
        self.workspaces = None
        self.tf_x = self.network.placeholder()
        self.tf_y = self.network.placeholder()
        self.decoded = self.network.define(self.tf_x)
        self.loss = tf.reduce_mean(tf.square(self.tf_y - self.decoded))
        self.train = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.datafile = data

        self.test_view_data_inputs = None
        self.test_view_data_targets = None
        self.test_view_costs = None
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
        self.test_view_data_inputs = \
            self.costmaps.test_inputs[:self.NUM_TEST]
        self.test_view_costs = self.workspaces[self.NUM_TRAIN: self.NUM_TRAIN +
                                                               self.NUM_TEST]

    def n_steps(self, n, begin=0):
        """ Do n steps of the algorithm over multiple environments """
        # initialize figure
        if self.DRAW or self.SAVE_TO_FILE:
            fig = plt.figure(figsize=(8, 5))
            grid = plt.GridSpec(5, self.NUM_TEST, wspace=0.4, hspace=0.3)

            a = [None] * 5
            for i in range(5):
                a[i] = [None] * self.NUM_TEST
                for j in range(self.NUM_TEST):
                    a[i][j] = fig.add_subplot(grid[i, j])

            for i in range(self.NUM_TEST):
                # occupancy grid
                a[0][i].imshow(self.test_view_data_inputs[i].reshape(self.PIXELS,
                                                                     self.PIXELS),
                               cmap=cm.magma)
                a[0][i].set_xticks(())
                a[0][i].set_yticks(())
                # original costmap
                a[1][i].imshow(self.test_view_costs[i].costmap, cmap=cm.magma)
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())

        for learch_step in range(begin, begin + n):
            print("learch step :", learch_step)
            self.costmaps.update_targets(self.costs, self.workspaces,
                                         self.learning, self._loss_scalar,
                                         self._loss_stddev, self._N)
            self.test_view_data_targets = \
                self.costmaps.test_targets[:self.NUM_TEST]
            if self.DRAW or self.SAVE_TO_FILE:
                for i in range(self.NUM_TEST):
                    # gradient
                    a[2][i].clear()
                    a[2][i].imshow(self.test_view_data_targets[i]
                                   .reshape(self.PIXELS, self.PIXELS), cmap=cm.magma)
                    a[2][i].set_xticks(())
                    a[2][i].set_yticks(())
                    # learned costs
                    a[4][i].clear()
                    a[4][i].imshow(np.log(self.costs[self.NUM_TRAIN + i]),
                                   cmap=cm.magma)
                    a[4][i].set_xticks(())
                    a[4][i].set_yticks(())

            self.sess.run(tf.global_variables_initializer())
            k = 0
            for step in range(self.BATCHES):
                b_x, b_y = self.costmaps.next_batch(self.BATCH_SIZE)
                _, decoded_, train_loss_ = self.sess.run(
                    [self.train, self.decoded, self.loss],
                    feed_dict={self.tf_x: self.network.resize_batch(b_x),
                               self.tf_y: self.network.resize_batch(b_y)})
                if step % 2 == 0:  # plotting
                    test_loss_ = self.sess.run(
                        self.loss,
                        {self.tf_x: self.network.resize_batch(
                            self.costmaps.test_inputs),
                            self.tf_y: self.network.resize_batch(
                                self.costmaps.test_targets)})
                    epoch = self.costmaps.epochs_completed
                    infostr = str()
                    infostr += 'step: {:8}, epoch: {:3}, '.format(step, epoch)
                    infostr += 'train loss: {:.4f}, test loss: {:.4f}'.format(
                        train_loss_, test_loss_)
                    print(infostr)

                    if self.DRAW or self.SAVE_TO_FILE:
                        decoded_data = self.sess.run(
                            self.decoded, {self.tf_x: self.network.
                                resize_batch(self.test_view_data_inputs)})
                        for i in range(self.NUM_TEST):
                            # learned gradient
                            a[3][i].clear()
                            a[3][i].imshow(self.network.resize_output(
                                decoded_data, i), cmap=cm.magma)
                            a[3][i].set_xticks(())
                            a[3][i].set_yticks(())
                        if self.SAVE_TO_FILE and (k % 20 == 0):
                            directory = learning_data_dir() + os.sep + "results"
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            fig.savefig(directory + os.sep + 'images_{:03}.pdf'
                                        .format(k))
                            plt.close(fig)
                        k += 1
                        if self.DRAW:
                            plt.draw()
                            plt.pause(0.01)
            if self.DRAW or self.SAVE_TO_FILE:
                plt.close(fig)
            # Update training costmaps
            decoded_data = self.sess.run(
                self.decoded, {self.tf_x: self.network.resize_batch(
                    self.costmaps.train_inputs)})
            self.decoded_data_train.append(decoded_data)
            # scale data
            v = ((np.max(decoded_data) - np.min(decoded_data)) / 2)
            decoded_data = (decoded_data - v)
            self.costs[:self.NUM_TRAIN] = self.costs[:self.NUM_TRAIN] \
                                          * np.exp(get_stepsize(learch_step,
                                                                self._learning_rate,
                                                                self._stepsize_scalar)
                                                   * decoded_data.reshape((-1,
                                                                           self.PIXELS,
                                                                           self.PIXELS)))
            # Update test costmaps
            decoded_data = self.sess.run(
                self.decoded, {self.tf_x: self.network.resize_batch(
                    self.costmaps.test_inputs)})
            self.decoded_data_test.append(decoded_data)
            decoded_data = (decoded_data - v)
            self.costs[self.NUM_TRAIN:] = self.costs[self.NUM_TRAIN:] \
                                          * np.exp(get_stepsize(learch_step,
                                                                self._learning_rate,
                                                                self._stepsize_scalar)
                                                   * decoded_data.reshape((-1,
                                                                           self.PIXELS,
                                                                           self.PIXELS)))
            self.learned_maps.append(np.log(self.costs))
            print("step size: ", get_stepsize(learch_step, self._learning_rate,
                                              self._stepsize_scalar))

        return np.log(self.costs), _, learch_step

    def solve(self, begin=0):
        """ Compute the algorithm for multiple environments
            until the weights converge
        """

        return
