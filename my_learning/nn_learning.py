import common_import

from my_utils.my_utils import *
from nn.nn_utils import *
import tensorflow as tf
from pyrieef.learning.tf_networks import *
from abc import abstractmethod
from my_utils.output_costmap import *


class NN_Learning():
    """
    """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        self.workspace = workspace
        # Parameters to compute the cost map from RBFs
        self.nb_points = nb_points
        self.sigma = sigma
        self.nb_rbfs = nb_rbfs

        self.w = np.exp(np.ones(nb_rbfs ** 2))
        self.instances = []

        # Parameters to compute the step size
        self._learning_rate = 1
        self._stepsize_scalar = 1

        self.convergence = 0.1

        self.network = ConvDeconvResize()
        self.tf_x = self.network.placeholder()
        self.decoded = self.network.define(self.tf_x)
        self.saver = tf.train.Saver()
        self.model = None

    @abstractmethod
    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        C = NN_Learning.NN_Learning_Instance(phi, paths, starts, targets,
                                             self.workspace)
        self.instances.append(C)

    @abstractmethod
    def n_steps(self, n, begin=0):  # TODO update
        """ Do n steps of the algorithm over multiple environments """
        for step in range(begin, begin + n):
            print("step :", step)
            input = []
            for j, i in enumerate(self.instances):
                i.update()
                input.append(i.get_input())
            input = np.array(input)

            with tf.Session() as sess:
                # Restore variables from disk.
                self.saver.restore(sess, home + '/../data/tf_models/'
                                   + self.model + '/model.ckpt')
                output = sess.run(self.decoded,
                                  feed_dict={self.tf_x: self.network.
                                  resize_batch(input)})
            for j, i in enumerate(self.instances):
                g = self.network.resize_output(output, j)
                # Exponentiated gradient descent
                i.costmap = i.costmap * \
                            np.exp(get_stepsize(step, self._learning_rate,
                                                self._stepsize_scalar) * g)
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))

        costmaps = []
        optimal_paths = []
        for j, i in enumerate(self.instances):
            # compute the learned costmaps and their optimal paths
            # for the learned weight
            costmaps.append(np.log(i.costmap))
            i.learned_maps.append(np.log(i.costmap))
            map = np.log(i.costmap) - np.amin(np.log(i.costmap))
            _, _, paths = plan_paths(len(i.sample_trajectories), map,
                                     self.workspace, starts=i.sample_starts,
                                     targets=i.sample_targets)
            optimal_paths.append(paths)
            i.optimal_paths.append(paths)
        return costmaps, optimal_paths, None, step

    def solve(self, begin=0):
        """ Compute the algorithm for multiple environments
            until the weights converge
        """
        step = begin
        e = 10
        w_old = copy.deepcopy(self.w)
        # Iterate until convergence
        while e > self.convergence:
            print("step :", step)
            input = []
            for j, i in enumerate(self.instances):
                i.update(self.w)
                input.append(i.get_input())
            input = np.array(input)

            with tf.Session() as sess:
                # Restore variables from disk.
                self.saver.restore(sess, home + '/../data/tf_models/'
                                   + self.model + '/model.ckpt')
                output = sess.run(self.decoded,
                                  feed_dict={self.tf_x: self.network.
                                  resize_batch(input)})

            w_t = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                g = self.network.resize_output(output, j)
                w_t += np.tensordot(i.phi, g)
                i.gradients.append(g)
                # show(g, self.workspace, 'SHOW')
                # Exponentiated gradient descent
                # i.costmap = i.costmap * \
                #            np.exp(get_stepsize(step, self._learning_rate,
                #                                self._stepsize_scalar) * g)
                # print(i.costmap.sum())
            self.w = self.w * np.exp(get_stepsize(step, self._learning_rate,
                                                  self._stepsize_scalar) *
                                     (w_t / len(self.instances)))
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
            e = np.amax(np.abs(self.w - w_old))
            w_old = copy.deepcopy(self.w)
            print("convergence: ", e)
            step += 1

        costmaps = []
        optimal_paths = []
        for j, i in enumerate(self.instances):
            # compute the learned costmaps and their optimal paths
            # for the learned weight
            costmap = get_costmap(i.phi, np.log(self.w))
            costmaps.append(costmap)
            i.learned_maps.append(costmap)
            map = costmap - np.amin(costmap)
            _, _, paths = plan_paths(len(i.sample_trajectories), map,
                                     self.workspace, starts=i.sample_starts,
                                     targets=i.sample_targets)
            optimal_paths.append(paths)
            i.optimal_paths.append(paths)
        return costmaps, optimal_paths, np.log(self.w), step

    class NN_Learning_Instance():
        """ Implements the new algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            # Parameters to compute the loss map
            self._loss_scalar = 1  # /len(paths)
            self._loss_stddev = 5  # 10

            self._N = 35  # 100

            self.workspace = workspace

            # Examples
            self.sample_trajectories = paths
            self.sample_starts = starts
            self.sample_targets = targets

            self.phi = phi
            self.costmap = np.exp(get_costmap(self.phi,
                                              np.ones(self.phi.shape[0])))

            self.loss_map = np.zeros((len(paths), phi.shape[1], phi.shape[2]))

            self.learned_maps = []
            self.optimal_paths = []
            self.gradients = []
            self.loss_agumented_maps = []
            self.loss_augmented_occupancy = []
            self.occupancy = []

        def update(self, w):
            self.w = w
            costmap = get_costmap(self.phi, self.w)
            self.costmap = costmap
            map = get_costmap(self.phi, np.log(self.w))
            self.learned_maps.append(map)
            _, _, paths = plan_paths(len(self.sample_trajectories), map,
                                     self.workspace, starts=self.sample_starts,
                                     targets=self.sample_targets)
            self.optimal_paths.append(paths)

        @abstractmethod
        def get_input(self):
            """ Compute one step of the new algorithm """
            map = get_avg_learch_occ_input()
            self.occupancy.append(map)
            return map


class NN_Learch(NN_Learning):
    """
    """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        NN_Learning.__init__(self, nb_points, nb_rbfs, sigma, workspace)

        # Parameters to compute the step size
        self._learning_rate = 1
        self._stepsize_scalar = 1

        self.convergence = 0.1

        self.model = 'learch_data_10'

    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        I = NN_Learch.NN_Learch_Instance(phi, paths, starts, targets,
                                         self.workspace)
        self.instances.append(I)

    class NN_Learch_Instance(NN_Learning.NN_Learning_Instance):
        """ Implements the algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            NN_Learning.NN_Learning_Instance.__init__(self, phi, paths, starts,
                                                      targets, workspace)
            # Parameters to compute the loss map
            self._loss_scalar = 1  # /len(paths)
            self._loss_stddev = 5  # 10

        def get_input(self):
            """ Compute one step of the new algorithm """
            map = get_learch_input(self.costmap, self.sample_trajectories,
                                   self.sample_starts, self.sample_targets,
                                   self._loss_stddev, self._loss_scalar)
            self.occupancy.append(map)
            return map


class NN_MaxEnt(NN_Learning):
    """
    """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        NN_Learning.__init__(self, nb_points, nb_rbfs, sigma, workspace)

        # Parameters to compute the step size
        self._learning_rate = 1
        self._stepsize_scalar = 1

        self.convergence = 0.1

        self.model = 'maxEnt_data_10'

    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        I = NN_MaxEnt.NN_MaxEnt_Instance(phi, paths, starts, targets,
                                         self.workspace)
        self.instances.append(I)

    class NN_MaxEnt_Instance(NN_Learning.NN_Learning_Instance):
        """ Implements the algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            NN_Learning.NN_Learning_Instance.__init__(self, phi, paths, starts,
                                                      targets, workspace)
            # Parameters to compute the loss map
            self._N = 35

        def get_input(self):
            """ Compute one step of the new algorithm """
            map = get_maxEnt_input(self.costmap, self._N,
                                   self.sample_trajectories,
                                   self.sample_targets, self.phi)
            self.occupancy.append(map)
            return map


class NN_Occ(NN_Learning):
    """
    """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        NN_Learning.__init__(self, nb_points, nb_rbfs, sigma, workspace)

        # Parameters to compute the step size
        self._learning_rate = 1
        self._stepsize_scalar = 1

        self.convergence = 0.1

        self.model = 'occ_data_10'

    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        I = NN_Occ.NN_Occ_Instance(phi, paths, starts, targets, self.workspace)
        self.instances.append(I)

    # def solve(self, begin=0):
    #    x = super().solve(begin)
    #    print("solve reached")
    #    return x

    class NN_Occ_Instance(NN_Learning.NN_Learning_Instance):
        """ Implements the algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            NN_Learning.NN_Learning_Instance.__init__(self, phi, paths, starts,
                                                      targets, workspace)
            # Parameters to compute the loss map
            self._N = 35

        def get_input(self):
            """ Compute one step of the new algorithm """
            map = get_occ_input(self.costmap, self._N,
                                self.sample_trajectories,
                                self.sample_targets)
            self.occupancy.append(map)
            return map


class NN_Loss_Aug_Occ(NN_Learning):
    """
    """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        NN_Learning.__init__(self, nb_points, nb_rbfs, sigma, workspace)

        # Parameters to compute the step size
        self._learning_rate = 1
        self._stepsize_scalar = 1

        self.convergence = 0.1

        self.model = 'loss_aug_occ_data_10'

    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        I = NN_Loss_Aug_Occ.NN_Loss_Aug_Occ_Instance(phi, paths, starts,
                                                     targets, self.workspace)
        self.instances.append(I)

    class NN_Loss_Aug_Occ_Instance(NN_Learning.NN_Learning_Instance):
        """ Implements the algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            NN_Learning.NN_Learning_Instance.__init__(self, phi, paths, starts,
                                                      targets, workspace)
            # Parameters to compute the loss map
            self._loss_scalar = 1  # /len(paths)
            self._loss_stddev = 5  # 10

            self._N = 35

        def get_input(self):
            """ Compute one step of the new algorithm """
            map = get_loss_aug_occ_input(self.costmap,
                                         self.sample_trajectories,
                                         self.sample_targets,
                                         self._loss_scalar,
                                         self._loss_stddev, self._N)
            self.occupancy.append(map)
            return map
