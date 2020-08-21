import common_import

from pyrieef.learning.inverse_optimal_control import *
from my_utils.my_utils import *
from my_utils.costmap import *


class Learch2D(Learch):
    """ Implements the LEARCH algorithm for a 2D squared map """

    def __init__(self, nb_points, centers, sigma,
                 paths, starts, targets, workspace):
        Learch.__init__(self, len(paths))

        self.workspace = workspace

        # Parameters to compute the loss map
        self._loss_scalar = 1.3
        self._loss_stddev = 10
        # Parameters to compute the step size
        self._learning_rate = 0.2
        self._stepsize_scalar = 2
        # Regularization parameters for the linear regression
        self._l2_regularizer = 1  # 1e-6
        self._proximal_regularizer = 0  # 1e-6
        # Change between gradient descent and exponentiated gradient descent
        self.exponentiated_gd = True

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
        self.costmap = np.zeros((nb_points, nb_points))
        self.loss_map = np.zeros((len(paths), nb_points, nb_points))
        self.D = np.empty((3, 0))
        self.policy = np.zeros((nb_points ** 2))
        self.visitation_frequency = np.zeros((nb_points, nb_points))
        self.transition_probability = np.zeros((nb_points ** 2 * 8,
                                                nb_points ** 2))

        # Data structures to save the progress of LEARCH in each iteration
        self.optimal_paths = []
        self.maps = []
        self.weights = []

        self.initialize_mydata()

    def initialize_mydata(self):
        """ Create the loss maps for each sample trajectory
            initialize the costmap with RBFs each of weight 1
        """
        for i, t in enumerate(self.sample_trajectories):
            self.loss_map[i] = scaled_hamming_loss_map(
                t, self.nb_points, self._loss_scalar, self._loss_stddev)
        self.costmap = get_costmap(
            self.nb_points, self.centers, self.sigma, self.w, self.workspace)

    def planning(self):
        """ Compute the optimal path for each start and
            target state in the expample trajectories
            Add the states to self.D where the cost function has to
            increase/decrease
        """
        maps = np.exp(self.costmap) - self.loss_map - \
               np.ones(self.costmap.shape) * \
               np.amin(np.exp(self.costmap) - self.loss_map)

        optimal_paths = plan_paths_fix_start(self.sample_starts,
                                             self.sample_targets, maps,
                                             self.workspace)

        for i, (trajectory, optimal_path) in enumerate(zip(self.sample_trajectories,
                                                           optimal_paths)):
            # Add the states of the optimal trajectory to D
            # The costs should be increased in the states
            # of the optimal trajectory
            x1 = np.asarray(np.transpose(optimal_path)[:][0])
            x2 = np.asarray(np.transpose(optimal_path)[:][1])
            D = np.vstack((x1, x2, np.ones(x1.shape)))
            self.D = np.hstack((self.D, D))

            # Add the states of the example trajectory to D
            # The costs should be decreased in the states
            # of the example trajectory
            x1 = np.asarray(np.transpose(trajectory)[:][0])
            x2 = np.asarray(np.transpose(trajectory)[:][1])
            D = np.vstack((x1, x2, - np.ones(x1.shape)))
            self.D = np.hstack((self.D, D))
        self.optimal_paths.append(optimal_paths)

    def supervised_learning(self, t):
        """ Train a regressor on D
            compute the new weights with gradient descent
        """
        C = self.D[:][2]
        x1 = self.D[:][0].astype(int)
        x2 = self.D[:][1].astype(int)
        Phi = self.phi[:, x1, x2].T

        if self.exponentiated_gd:
            # Exponentiated Gradient Descent
            w_new = linear_regression(Phi, C, np.exp(self.w), self._l2_regularizer,
                                      self._proximal_regularizer)
            # w_new = self.get_subgadient()
            self.w = np.exp(self.w * get_stepsize(t, self._learning_rate,
                                                  self._stepsize_scalar) * w_new)
            self.w = np.log(self.w)
        else:
            # Gradient Descent
            w_new = linear_regression(Phi, C, self.w, self._l2_regularizer,
                                      self._proximal_regularizer)
            # w_new = self.get_subgadient()
            self.w = self.w + get_stepsize(t, self._learning_rate,
                                           self._stepsize_scalar) * w_new
        self.weights.append(self.w)
        self.costmap = get_costmap(
            self.nb_points, self.centers, self.sigma, self.w, self.workspace)

    def get_subgadient(self):
        """ Return the subgradient of the maximum margin planning objective """
        g = 0
        for i, op in enumerate(self.optimal_paths[-1]):
            g += np.sum(self.phi[:, np.asarray(self.sample_trajectories[i]).T[:][0],
                        np.asarray(self.sample_trajectories[i]).T[:][1]], axis=1) \
                 - np.sum(self.phi[:, np.asarray(op).T[:][0],
                          np.asarray(op).T[:][1]], axis=1)
        g = - g / (len(self.sample_trajectories)) + self._l2_regularizer * self.w
        return g

    def one_step(self, t):
        """ Compute one step of the LEARCH algorithm """
        time_0 = time.time()
        print("step :", t)
        self.planning()
        self.supervised_learning(t)
        print("took t : {} sec.".format(time.time() - time_0))
        self.maps.append(self.costmap)
        if self.exponentiated_gd:
            return self.maps, self.optimal_paths, np.exp(self.weights)
        else:
            return self.maps, self.optimal_paths, self.weights

    def n_step(self, n):
        """ Compute n steps of the LEARCH algorithm """
        for i in range(n):
            self.one_step(i)
        if self.exponentiated_gd:
            return self.maps, self.optimal_paths, np.exp(self.weights)
        else:
            return self.maps, self.optimal_paths, self.weights

    def solve(self):
        """ Compute LEARCH until the weights converge """
        w_old = copy.deepcopy(self.w)
        e = 10
        i = 0
        while e > 1:
            self.one_step(i)
            # print("w: ", self.w)
            e = (np.absolute(self.w - w_old)).sum()
            print("convergence: ", e)
            w_old = copy.deepcopy(self.w)
            i += 1
        if self.exponentiated_gd:
            return self.maps, self.optimal_paths, np.exp(self.weights)
        else:
            return self.maps, self.optimal_paths, self.weights