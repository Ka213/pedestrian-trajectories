from common_import import *

from my_utils.my_utils import *
from my_utils.environment import *
from my_learning.irl import *


class MaxEnt(Learning):
    """ Implements the maximum entropy approach for a 2D squared map """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        Learning.__init__(self, nb_points, nb_rbfs, sigma, workspace)

        # Parameters to compute the step size
        self._learning_rate = 0.4
        self._stepsize_scalar = 1

        self.convergence = 1

        self.weights = []
        self.w = np.zeros(nb_rbfs ** 2)

    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        M = self.MaxEnt_instance(phi, paths, starts, targets, self.workspace)
        self.instances.append(M)

    def n_steps(self, n, begin=0):
        """ Do n steps of the maxEnt algorithm over multiple environments
            using gradient descent
        """
        for step in range(begin, begin + n):
            print("step :", step)
            # Average the gradient over multiple environments
            g = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.update(self.w)
                gradient = i.get_gradient()
                g += gradient
            # Gradient descent step
            self.w = self.w + get_stepsize(step, self._learning_rate,
                                           self._stepsize_scalar) * \
                     (g / len(self.instances))
            self.weights.append(self.w)
            # print("w ", self.w , self.w.sum())
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
        # Compute the learned costmaps and example paths
        costmaps = []
        ex_paths = []
        for _, i in enumerate(self.instances):
            i.update(self.w)
            costmaps.append(i.costmap)
            ex_paths.append(i.optimal_paths[-1])
        return costmaps, ex_paths, self.w, step

    def solve(self, begin=0):
        """ Compute the maxEnt over multiple environments
            until the weights converge using gradient descent
        """
        step = begin
        w_old = copy.deepcopy(self.w)
        e = 10
        while e > self.convergence:
            print("step :", step)
            # Average gradient over multiple environments
            g = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.update(self.w)
                gradient = i.get_gradient()
                g += gradient
            g = g / len(self.instances)
            # g = g - g.min()
            self.w = self.w + get_stepsize(step, self._learning_rate,
                                           self._stepsize_scalar) * g
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
            e = np.max(np.abs(self.w - w_old))
            print("convergence: ", e)
            w_old = copy.deepcopy(self.w)
            step += 1
        # Compute the learned costmaps and example paths
        costmaps = []
        ex_paths = []
        for _, i in enumerate(self.instances):
            i.update(self.w)
            costmaps.append(i.learned_maps[-1])
            ex_paths.append(i.optimal_paths[-1])
        return costmaps, ex_paths, self.w, step

    class MaxEnt_instance(Learning.Instance):
        """ Implements the maxEnt algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            Learning.Instance.__init__(self, phi, paths, starts, targets,
                                       workspace)

            self._N = 45

            self.w = np.zeros(phi.shape[0])
            self.transition_probability = \
                get_transition_probabilities(self.costmap)
            self.f_empirical = get_empirical_feature_count \
                (self.sample_trajectories, self.phi)
            self.d = []
            self.f_expected = []
            self.f = []

        def update(self, w):
            """ Update the weights and the costmap """
            self.w = w
            map = get_costmap(self.phi, self.w)
            self.costmap = map
            self.learned_maps.append(map)
            _, _, p = plan_paths(len(self.sample_trajectories), self.costmap,
                                 self.workspace, self.sample_starts,
                                 self.sample_targets)
            self.optimal_paths.append(p)

        def get_gradient(self):
            """ Compute the gradient of the maximum entropy objective """
            # Get empirical feature counts
            f_empirical = self.f_empirical

            # Calculate the learners expected state frequency
            try:
                d = get_expected_edge_frequency(self.costmap, self._N,
                                                self.phi.shape[1],
                                                self.sample_starts,
                                                self.sample_targets,
                                                self.workspace)
                self.d.append(d)
            except Exception:
                raise
            f_expected = np.tensordot(self.phi, d)
            self.f_expected.append(get_costmap(self.phi, f_expected))
            f = f_empirical - f_expected
            # Convert since we use costs not rewards
            f = - f - np.min(- f)
            self.f.append(get_costmap(self.phi, f))
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
