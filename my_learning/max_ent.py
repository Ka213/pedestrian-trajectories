from common_import import *

from my_utils.my_utils import *
from my_utils.environment import *
from my_learning.irl import *


class MaxEnt(Learning):
    """ Implements the Maximum Entropy approach for a 2D squared map """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        Learning.__init__(self, nb_points, nb_rbfs, sigma, workspace)

        # Parameters to compute the step size
        self._learning_rate = 0.114
        self._stepsize_scalar = 20

        self.convergence = 0.01

        self.w = np.exp(np.zeros(nb_rbfs ** 2))

    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the maxEnt computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        M = self.MaxEnt_instance(phi, paths, starts, targets, self.workspace)
        self.instances.append(M)

    def n_steps(self, n, begin=0):
        """ Do n steps of the maxEnt algorithm over multiple environments
            using exponentiated gradient descent
        """
        for step in range(begin, begin + n):
            print("step :", step)
            g = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.update(np.log(self.w))
                gradient = i.get_gradient()
                g += gradient
            self.w = self.w * np.exp(get_stepsize(step, self._learning_rate,
                                                  self._stepsize_scalar) * \
                                     (g / len(self.instances)))
            print("step size: ", np.exp(get_stepsize(step, self._learning_rate,
                                                     self._stepsize_scalar)))
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            i.update(np.log(self.w))
            costmaps.append(i.costmap)
            optimal_paths.append(i.optimal_paths[-1])
        return costmaps, optimal_paths, np.log(self.w), step

    def solve(self, begin=0):
        """ Compute the maxEnt over multiple environments
            until the weights converge
            using exponentiated gradient descent
        """
        step = begin
        w_old = copy.deepcopy(self.w)
        e = 10
        while e > 0.1:
            print("step :", step)
            g = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.update(np.log(self.w))
                gradient = i.get_gradient()
                g += gradient
            self.w = self.w * np.exp(get_stepsize(step, self._learning_rate,
                                                  self._stepsize_scalar) * \
                                     (g / len(self.instances)))
            print("step size: ", np.exp(get_stepsize(step, self._learning_rate,
                                                     self._stepsize_scalar)))
            e = np.amax(self.w - w_old).sum()
            print("convergence: ", e)
            w_old = copy.deepcopy(self.w)
            step += 1
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            i.update(np.log(self.w))
            costmaps.append(i.costmap)
            optimal_paths.append(i.optimal_paths[-1])
        return costmaps, optimal_paths, np.log(self.w), step

    def n_steps_gd(self, n, begin=0):
        """ Do n steps of the maxEnt algorithm over multiple environments
            using gradient descent
        """
        self.w = np.log(self.w)
        for step in range(begin, begin + n):
            print("step :", step)
            g = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.update(self.w)
                gradient = i.get_gradient()
                g += gradient
            self.w = self.w + get_stepsize(step, self._learning_rate,
                                           self._stepsize_scalar) * \
                     (g / len(self.instances))
            # print("w ", self.w , self.w.sum())
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            i.update(self.w)
            costmaps.append(i.costmap)
            optimal_paths.append(i.optimal_paths[-1])
        return costmaps, optimal_paths, self.w, step

    def solve_gd(self, begin=0):
        """ Compute the maxEnt over multiple environments
            until the weights converge
            using gradient descent
        """
        self.w = np.log(self.w)
        step = begin
        w_old = copy.deepcopy(self.w)
        e = 10
        while e > self.convergence:
            print("step :", step)
            g = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.update(self.w)
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
            i.update(self.w)
            costmaps.append(i.costmap)
            optimal_paths.append(i.optimal_paths[-1])
        return costmaps, optimal_paths, self.w, step

    class MaxEnt_instance(Learning.Instance):
        """ Implements the maxEnt algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            Learning.Instance.__init__(self, phi, paths, starts, targets,
                                       workspace)

            self._N = 185

            self.w = np.zeros(phi.shape[0])
            self.transition_probability = \
                get_transition_probabilities(self.costmap, phi.shape[1])
            self.f_empirical = get_empirical_feature_count \
                (self.sample_trajectories, self.phi)

        def update(self, w):
            """ Update the weights and the costmap """
            self.w = w
            map = np.tensordot(self.w, self.phi, axes=1)
            self.costmap = map
            self.learned_maps.append(map)
            map = self.costmap - np.amin(self.costmap)
            _, _, op = plan_paths(len(self.sample_trajectories), map,
                                  self.workspace, self.sample_starts,
                                  self.sample_targets)
            self.optimal_paths.append(op)

        def get_gradient(self):
            """ Compute the gradient of the maximum entropy objective """
            # Get expected empirical feature counts
            f_empirical = self.f_empirical
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
    loss = np.zeros(len(learned_maps))
    for i, (map, demo) in enumerate(zip(learned_maps, demonstrations)):
        for _, d in enumerate(demo):
            loss[i] += np.sum(map[np.asarray(d)[:, 0], np.asarray(d)[:, 1]])
    loss = (loss / nb_samples)
    if w is not None:
        loss += np.linalg.norm(w)
    return loss
