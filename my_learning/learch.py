import common_import

from pyrieef.learning.inverse_optimal_control import *
from my_utils.my_utils import *
from my_utils.environment import *


class Learch2D():
    """ Implements the LEARCH algorithm for a 2D squared map """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        self.workspace = workspace
        # Parameters to compute the cost map from RBFs
        self.nb_points = nb_points
        self.sigma = sigma
        self.nb_rbfs = nb_rbfs
        # Parameters to compute the step size
        self._learning_rate = 1
        self._stepsize_scalar = 1

        self.convergence = 0.1

        self.w = np.ones(nb_rbfs ** 2)
        self.instances = []

    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the LEARCH computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        L = self.Learch_instance(phi, paths, starts, targets, self.workspace)
        self.instances.append(L)

    def get_regularization(self):
        """ Return the regularization factors to compute the loss """
        l2 = 0
        l_proximal = 0
        if len(self.instances) > 0:
            l = self.instances[0]
            l2 = l._l2_regularizer
            l_proximal = l._proximal_regularizer
        return l2, l_proximal

    def n_step(self, n):
        """ Do n steps of the LEARCH algorithm over multiple environments """
        step = 0
        for i in range(n):
            print("step :", step)
            w_t = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.w = self.w
                i.costmap = np.tensordot(np.exp(self.w), i.phi, axes=1)
                w = i.one_step(step)
                w_t += w
            self.w = np.exp(self.w + get_stepsize(step, self._learning_rate,
                                                  self._stepsize_scalar) *
                            (w_t / len(self.instances)))
            # print("w ", self.w , self.w.sum())
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
            self.w = np.log(self.w)
            step += 1
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            costmap = np.tensordot(self.w, i.phi, axes=1)
            costmaps.append(costmap)
            map = costmap - np.amin(costmap)
            _, _, paths = plan_paths(len(i.sample_trajectories), map,
                                     self.workspace, starts=i.sample_starts,
                                     targets=i.sample_targets)
            optimal_paths.append(paths)
        return costmaps, optimal_paths, np.exp(self.w), step

    def solve(self):
        """ Compute the LEARCH over multiple environments
            until the weights converge
        """
        step = 0
        w_old = copy.deepcopy(self.w)
        e = 10
        # Iterate until convergence
        while e > self.convergence:
            print("step :", step)
            w_t = np.zeros(self.nb_rbfs ** 2)
            for j, i in enumerate(self.instances):
                i.w = self.w
                i.costmap = np.tensordot(np.exp(self.w), i.phi, axes=1)
                w = i.one_step(step)
                w_t += w
            # Gradient descent rule of the LEARCH algorithm
            self.w = np.exp(self.w + get_stepsize(step, self._learning_rate,
                                                  self._stepsize_scalar) *
                            (w_t / len(self.instances)))
            print("step size: ", get_stepsize(step, self._learning_rate,
                                              self._stepsize_scalar))
            self.w = np.log(self.w)
            e = np.amax(self.w - w_old).sum()
            print("convergence: ", e)
            w_old = copy.deepcopy(self.w)
            step += 1
        # compute the learned costmaps and their optimal paths
        # for the learned weight w
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            costmap = np.tensordot(self.w, i.phi, axes=1)
            costmaps.append(costmap)
            map = costmap - np.amin(costmap)
            _, _, paths = plan_paths(len(i.sample_trajectories), map,
                                     self.workspace, starts=i.sample_starts,
                                     targets=i.sample_targets)
            optimal_paths.append(paths)
        return costmaps, optimal_paths, np.exp(self.w), step

    class Learch_instance(Learch):
        """ Implements the LEARCH algorithm for one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            Learch.__init__(self, len(paths))
            self.workspace = workspace

            # Parameters to compute the loss map
            self._loss_scalar = 1
            self._loss_stddev = 10
            # Regularization parameters for the linear regression
            self._l2_regularizer = 0.01  # 1e-6
            self._proximal_regularizer = 0  # 1e-6

            # Examples
            self.sample_trajectories = paths
            self.sample_starts = starts
            self.sample_targets = targets

            self.phi = phi
            self.w = np.log(np.ones(phi.shape[0]))
            self.costmap = np.tensordot(self.w, self.phi, axes=1)
            self.loss_map = np.zeros((len(paths), phi.shape[1], phi.shape[2]))

            self.create_loss_maps()

        def create_loss_maps(self):
            """ Create the loss maps for each sample trajectory """
            for i, t in enumerate(self.sample_trajectories):
                self.loss_map[i] = scaled_hamming_loss_map(
                    t, self.phi.shape[1], self._loss_scalar, self._loss_stddev)

        def planning(self):
            """ Compute the optimal path for each start and
                target state in the expample trajectories
                Add the states to the set D where the cost function has to
                increase/decrease
            """
            D = np.empty((3, 0))
            optimal_paths = []
            for i, (s, t) in enumerate(zip(self.sample_starts, self.sample_targets)):
                map = self.costmap - self.loss_map[i] - \
                      np.ones(self.costmap.shape) * \
                      np.amin(self.costmap - self.loss_map[i])
                _, _, paths = plan_paths(1, map, self.workspace, starts=[s],
                                         targets=[t])
                optimal_paths.append(paths[0])

            for i, (trajectory, optimal_path) in enumerate(zip(
                    self.sample_trajectories, optimal_paths)):
                # Add the states of the optimal trajectory to D
                # The costs should be increased in the states
                # of the optimal trajectory
                x1 = np.asarray(np.transpose(optimal_path)[:][0])
                x2 = np.asarray(np.transpose(optimal_path)[:][1])
                d = np.vstack((x1, x2, np.ones(x1.shape)))
                D = np.hstack((D, d))

                # Add the states of the example trajectory to D
                # The costs should be decreased in the states
                # of the example trajectory
                x1 = np.asarray(np.transpose(trajectory)[:][0])
                x2 = np.asarray(np.transpose(trajectory)[:][1])
                d = np.vstack((x1, x2, - np.ones(x1.shape)))
                D = np.hstack((D, d))
            return D, optimal_paths

        def supervised_learning(self, t, D):
            """ Train a regressor on D
                compute the hypothesis of the new weights with linear regression
            """
            C = D[:][2]
            x1 = D[:][0].astype(int)
            x2 = D[:][1].astype(int)
            Phi = self.phi[:, x1, x2].T

            w_new = linear_regression(Phi, C, np.exp(self.w),
                                      self._l2_regularizer,
                                      self._proximal_regularizer)
            return w_new

        def get_subgadient(self, optimal_paths):
            """ Return the subgradient of the maximum margin planning objective
            """
            g = 0
            for i, op in enumerate(optimal_paths):
                g += np.sum(
                    self.phi[:, np.asarray(self.sample_trajectories[i]).T[:][0],
                    np.asarray(self.sample_trajectories[i]).T[:][1]], axis=1) \
                     - np.sum(self.phi[:, np.asarray(op).T[:][0],
                              np.asarray(op).T[:][1]], axis=1)
            g = - g / (len(self.sample_trajectories)) \
                + self._l2_regularizer * self.w
            return g

        def one_step(self, t):
            """ Compute one step of the LEARCH algorithm """
            d, optimal_paths = self.planning()
            w_new = self.supervised_learning(t, d)
            return w_new


def get_learch_loss(original_costmaps, optimal_paths, demonstrations, nb_samples,
                    l_2=None, l_proximal=None, w=None):
    """ Returns the LEARCH loss with or without regularization factor """
    loss = np.zeros(len(original_costmaps))
    for i, (map, demo, path) in enumerate(zip(original_costmaps, demonstrations,
                                              optimal_paths)):
        for op, d in zip(path, demo):
            loss[i] += np.sum(map[np.asarray(d)[:, 0], np.asarray(d)[:, 1]]) \
                       - np.sum(map[np.asarray(op)[:, 0], np.asarray(op)[:, 1]])
    loss = (loss / nb_samples)
    if l_2 is not None and l_proximal is not None and w is not None:
        loss += (l_2 + l_proximal) * np.linalg.norm(w)
    return loss
