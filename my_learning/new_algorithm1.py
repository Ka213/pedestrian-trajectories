import common_import

from my_utils.my_utils import *
from my_utils.environment import *
from my_learning.irl import *


class NewAlgorithm_1(Learning):
    """ Implements the combined learning of the algorithms
        LEARCH and maximum Entropy
    """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        Learning.__init__(self, nb_points, nb_rbfs, sigma, workspace)

        # Parameters to compute the step size
        self._learning_rate = 1
        self._stepsize_scalar = 1

        self.convergence = 0.1

        self.w = np.exp(np.ones(nb_rbfs ** 2))

    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the LEARCH computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        C = NewAlgorithm_1.Instance(phi, paths, starts, targets, self.workspace)
        self.instances.append(C)

    def get_regularization(self):
        """ Return the regularization factors to compute the loss """
        l2 = 0
        l_proximal = 0
        if len(self.instances) > 0:
            l = self.instances[0]
            l2 = l._l2_regularizer
            l_proximal = l._proximal_regularizer
        return l2, l_proximal

    def n_steps(self, n, begin=0):
        """ Do n steps of the new algorithm over multiple environments """
        for step in range(begin, begin + n):
            print("step :", step)
            w_t = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.update(self.w)
                w = i.get_gradient()
                w_t += w
            w_t = w_t / w_t.sum()
            # Gradient descent rule of the LEARCH algorithm
            self.w = self.w * np.exp(get_stepsize(step, self._learning_rate,
                                                  self._stepsize_scalar) * w_t)
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
        # compute the learned costmaps and their optimal paths
        # for the learned weight w
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
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

    def solve(self, begin=0):
        """ Compute the new algorithm over multiple environments
            until the weights converge
        """
        step = begin
        w_old = copy.deepcopy(self.w)
        e = 10
        # Iterate until convergence
        while e > self.convergence:
            print("step :", step)
            w_t = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.update(self.w)
                w = i.get_gradient()
                w_t += w
            w_t = w_t / len(self.instances)
            print(w_t.sum())
            w_t = w_t / w_t.sum()
            # Exponentiated gradient descent
            self.w = self.w * np.exp(get_stepsize(step, self._learning_rate,
                                                  self._stepsize_scalar) * w_t)
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
            e = np.amax(np.abs(self.w - w_old))
            print("convergence: ", e)
            w_old = copy.deepcopy(self.w)
            step += 1
        # compute the learned costmaps and their optimal paths
        # for the learned weight w
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
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

    class Instance(Learning.Instance):
        """ Implements the new algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            Learning.Instance.__init__(self, phi, paths, starts, targets,
                                       workspace)
            # Regularization parameters for the linear regression
            self._l2_regularizer = 1
            self._proximal_regularizer = 0

            self._N = 150

            self.transition_probability = \
                get_transition_probabilities(self.costmap)

            self.weights = []


        def planning(self):
            """ Compute the optimal path for each start and
                target state in the expample trajectories
                Add the states to the set d where the cost function has to
                increase/decrease
            """

            try:
                o = get_expected_edge_frequency(self.transition_probability,
                                                self.costmap, self._N,
                                                self.phi.shape[1],
                                                self.sample_targets,
                                                self.sample_trajectories,
                                                self.workspace)
            except Exception:
                raise

            # Add the states of the optimal trajectory to d
            # The costs should be increased in the states
            # of the optimal trajectory
            X = np.arange(self.phi.shape[1])
            Y = np.arange(self.phi.shape[2])
            x1, x2 = np.meshgrid(X, Y)
            d1 = np.vstack((x1.flatten(), x2.flatten(), o.flatten()))

            d3 = np.empty((3, 0))
            for i, trajectory in enumerate(self.sample_trajectories):
                # Add the states of the example trajectory to d
                # The costs should be decreased in the states
                # of the example trajectory
                x1 = np.asarray(np.transpose(trajectory)[:][0])
                x2 = np.asarray(np.transpose(trajectory)[:][1])
                d_ = np.vstack((x1, x2, - np.ones(x1.shape)))
                d3 = np.hstack((d3, d_))

            d1[2] = d1[2] / d1[2].sum() * -d3[2].sum()
            d = np.hstack((d1, d3))
            return d

        def supervised_learning(self, d):
            """ Train a regressor on D
                compute the hypothesis of the new weights with linear regression
            """
            c = d[:][2]
            x1 = d[:][0].astype(int)
            x2 = d[:][1].astype(int)
            phi = self.phi[:, x1, x2].T

            w_new = linear_regression(phi, c, self.w,
                                      self._l2_regularizer,
                                      self._proximal_regularizer)
            self.weights.append(w_new)
            return w_new

        def get_gradient(self):
            """ Compute one step of the new algorithm """
            d = self.planning()
            w_new = self.supervised_learning(d)
            return w_new
