import common_import

from pyrieef.geometry.interpolation import *
from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *
from pyrieef.learning.inverse_optimal_control import *
from my_utils.my_utils import *
from costmap.costmap import *


class MaxEnt():
    """ Implements the Maximum Entropy approach for a 2D squared map """

    def __init__(self, nb_points, centers, sigma,
                 paths, starts, targets, workspace):

        self.workspace = workspace

        # Parameters to compute the step size
        self._learning_rate = 0.8
        self._stepsize_scalar = 1

        self._N = 10

        # Parameters to compute the cost map from RBFs
        self.nb_points = nb_points
        self.centers = centers
        self.sigma = sigma
        self.phi = get_phi(nb_points, centers, sigma, workspace)

        # Examples
        self.sample_trajectories = paths
        self.sample_starts = starts
        self.sample_targets = targets

        self.w = np.ones(len(centers))
        self.costmap = get_costmap(
            self.nb_points, self.centers, self.sigma, self.w, self.workspace)
        self.transition_probability = \
            get_transition_probabilities(self.costmap, self.nb_points)
        self.visitation_frequency = \
            get_expected_edge_frequency(self.transition_probability,
                                        self.costmap, self._N, self.nb_points)

        # Data structures to save the progress of MaxEnt in each iteration
        self.maps = []
        self.weights = []


    def one_step(self, t):
        """ Compute one gradient descent step """
        time_0 = time.time()
        print("step :", t)

        self.w = self.w + get_stepsize(t, self._learning_rate,
                                       self._stepsize_scalar) \
                 * self.get_gradient()
        self.visitation_frequency = \
            get_expected_edge_frequency(self.transition_probability,
                                        self.costmap, self._N, self.nb_points)
        print("took t : {} sec.".format(time.time() - time_0))
        self.maps.append(self.costmap)
        self.weights.append(self.w)
        return self.maps, self.weights

    def n_step(self, n):
        """ Compute n gradient descent step """
        for i in range(n):
            self.one_step(i)
        return self.maps, self.weights

    def solve(self):
        """ Compute gradient descent until convergence """
        w_old = copy.deepcopy(self.w)
        e = 10
        i = 0
        while e > 1:
            self.one_step(i)
            # print("w: ", self.w)
            e = (np.absolute(self.w - w_old)).sum()
            # print("convergence: ", e)
            w_old = copy.deepcopy(self.w)
            i += 1
        return self.maps, self.weights

    def get_gradient(self):
        """ Compute the gradient of the maximum entropy objective """
        Phi = get_phi(self.nb_points, self.centers, self.sigma, self.workspace)
        x_1 = np.empty(0)
        x_2 = np.empty(0)
        for i, trajectory in enumerate(self.sample_trajectories):
            x_1 = np.hstack((x_1, np.asarray(trajectory)[:, 0]))
            x_2 = np.hstack((x_2, np.asarray(trajectory)[:, 1]))
        f_empirical = np.sum(Phi[:, x_1.astype(int), x_2.astype(int)]) \
                      / len(self.sample_trajectories)

        a = f_empirical - np.sum(np.tile(
            self.visitation_frequency.reshape((self.nb_points, self.nb_points)),
            (len(self.w), 1, 1)) * Phi, axis=(1, 2))
        return a
