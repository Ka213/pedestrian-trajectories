from common_import import *

from my_utils.my_utils import *
from my_utils.environment import *


class MaxEnt():
    """ Implements the Maximum Entropy approach for a 2D squared map """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        self.workspace = workspace
        # Parameters to compute the cost map from RBFs
        self.nb_points = nb_points
        self.sigma = sigma
        self.nb_rbfs = nb_rbfs
        # Parameters to compute the step size
        self._learning_rate = 0.114
        self._stepsize_scalar = 20

        self.convergence = 0.01

        self.w = np.zeros(nb_rbfs ** 2)
        self.instances = []

    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the maxEnt computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        M = self.MaxEnt_instance(phi, paths, starts, targets, self.workspace)
        self.instances.append(M)

    def n_steps(self, n):
        """ Do n steps of the maxEnt algorithm over multiple environments """
        step = 0
        for i in range(n):
            print("step :", step)
            g = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.w = self.w
                i.costmap = np.tensordot(self.w, i.phi, axes=1)
                gradient = i.get_gradient()
                g += gradient
            self.w = self.w + get_stepsize(step, self._learning_rate,
                                           self._stepsize_scalar) * \
                     (g / len(self.instances))
            # print("w ", self.w , self.w.sum())
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
            step += 1
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            costmap = np.tensordot(self.w, i.phi, axes=1)
            costmaps.append(costmap)
            _, _, paths = plan_paths(len(i.sample_trajectories), costmap,
                                     self.workspace, starts=i.sample_starts,
                                     targets=i.sample_targets)
            optimal_paths.append(paths)
        return costmaps, optimal_paths, self.w, step

    def solve(self):
        """ Compute the maxEnt over multiple environments
            until the weights converge
        """
        step = 0
        w_old = copy.deepcopy(self.w)
        e = 10
        while e > self.convergence:
            print("step :", step)
            g = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.w = self.w
                i.costmap = np.tensordot(self.w, i.phi, axes=1)
                gradient = i.get_gradient()
                g += gradient
            self.w = self.w + get_stepsize(step, self._learning_rate,
                                           self._stepsize_scalar) * \
                     (g / len(self.instances))
            # print("w ", self.w , self.w.sum())
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
            e = np.amax(self.w - w_old).sum()
            print("convergence: ", e)
            w_old = copy.deepcopy(self.w)
            step += 1
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            costmap = np.tensordot(self.w, i.phi, axes=1)
            costmaps.append(costmap)
            _, _, paths = plan_paths(len(i.sample_trajectories), costmap,
                                     self.workspace, starts=i.sample_starts,
                                     targets=i.sample_targets)
            optimal_paths.append(paths)
        return costmaps, optimal_paths, self.w, step

    class MaxEnt_instance():
        """ Implements the maxEnt algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            self.workspace = workspace

            self._N = 185

            # Examples
            self.sample_trajectories = paths
            self.sample_starts = starts
            self.sample_targets = targets

            self.phi = phi
            self.w = np.zeros(phi.shape[0])
            self.costmap = np.tensordot(self.w, self.phi, axes=1)
            self.transition_probability = \
                get_transition_probabilities(self.costmap, phi.shape[1])

        def get_gradient(self):
            """ Compute the gradient of the maximum entropy objective """
            # Calculate expected empirical feature counts
            f_empirical = get_empirical_feature_count \
                (self.sample_trajectories, self.phi)
            # Calculate the learners expected feature count
            try:
                D = get_expected_edge_frequency(self.transition_probability,
                                                self.costmap, self._N,
                                                self.phi.shape[1],
                                                self.sample_targets,
                                                self.sample_trajectories,
                                                self.workspace)
            except Exception:
                raise
            f_expected = np.tensordot(self.phi, D)
            f = f_empirical - f_expected
            f = - f - np.min(- f)
            return f


def get_maxEnt_loss(learned_maps, demonstrations, nb_samples, w=None):
    """ Returns the maxEnt loss with or without regularization factor """
    loss = 0
    for map, demo in zip(learned_maps, demonstrations):
        for _, d in enumerate(demo):
            loss += np.sum(map[np.asarray(d)[:, 0], np.asarray(d)[:, 1]])
    loss = (loss / nb_samples)
    if w is not None:
        loss += np.linalg.norm(w)
    return loss
