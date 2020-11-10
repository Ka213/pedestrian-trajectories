import common_import

from abc import abstractmethod
from my_utils.environment import *
from my_utils.output_costmap import *


class Learning():
    """ Implements the abstract methods for the learning learning algorithms
        for multiple environments
        returns always the the one vecotor as weight
    """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        self.workspace = workspace
        # Parameters to compute the cost map from RBFs
        self.nb_points = nb_points
        self.sigma = sigma
        self.nb_rbfs = nb_rbfs

        self.w = np.exp(np.ones(nb_rbfs ** 2))
        self.instances = []

    @abstractmethod
    def add_environment(self, centers, paths, starts, targets):
        """ Add an new environment to the computation """
        phi = get_phi(self.nb_points, centers, self.sigma, self.workspace)
        I = self.Instance(phi, paths, starts, targets, self.workspace)
        self.instances.append(I)

    @abstractmethod
    def n_steps(self, n, begin=0):
        """ Returns a one vector as weights """
        # compute the learned costmaps and their optimal paths
        # for the weight w
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            costmap = get_costmap(i.phi, np.log(self.w))
            costmaps.append(costmap)
            i.learned_maps.append(costmap)
            _, _, paths = plan_paths(len(i.sample_trajectories), costmap,
                                     self.workspace, starts=i.sample_starts,
                                     targets=i.sample_targets)
            optimal_paths.append(paths)
            i.optimal_paths.append(paths)
        return costmaps, optimal_paths, np.log(self.w), n

    @abstractmethod
    def solve(self, begin=0):
        """ Returns a one vector as weights """
        # for _, i in enumerate(self.instances):
        #    i.update(self.w)

        # compute the learned costmaps and their optimal paths
        # for the weight w
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            costmap = get_costmap(i.phi, np.log(self.w))
            costmaps.append(costmap)
            i.learned_maps.append(costmap)
            _, _, paths = plan_paths(len(i.sample_trajectories), costmap,
                                     self.workspace, starts=i.sample_starts,
                                     targets=i.sample_targets)
            optimal_paths.append(paths)
            i.optimal_paths.append(paths)
        return costmaps, optimal_paths, np.log(self.w), 0

    class Instance():
        """ Implements the learning of one environment """

        def __init__(self, phi, paths, starts, targets, workspace):
            self.workspace = workspace

            # Examples
            self.sample_trajectories = paths
            self.sample_starts = starts
            self.sample_targets = targets

            self.phi = phi
            self.w = np.exp(np.ones(self.phi.shape[0]))
            self.costmap = get_costmap(self.phi, np.log(self.w))

            self.learned_maps = []
            self.optimal_paths = []

        @abstractmethod
        def update(self, w):
            """ Update the weights and the costmap """
            self.w = w
            map = get_costmap(self.phi, self.w)
            self.costmap = map
            map = get_costmap(self.phi, np.log(self.w))
            self.learned_maps.append(map)

            _, _, paths = plan_paths(len(self.sample_trajectories), map,
                                     self.workspace, starts=self.sample_starts,
                                     targets=self.sample_targets)
            self.optimal_paths.append(paths)

        @abstractmethod
        def get_gradient(self):
            return 0
