import common_import

from my_utils.my_utils import *
from my_utils.environment import *
from my_learning.irl import *


class Random(Learning):
    """ Implements the LEARCH algorithm for a 2D squared map """

    def __init__(self, nb_points, nb_rbfs, sigma, workspace):
        Learning.__init__(self, nb_points, nb_rbfs, sigma, workspace)

        self.w = np.random.random(self.nb_rbfs ** 2)

    def n_steps(self, n, begin=0):
        """ Returns a random vector as weights """
        # compute the learned costmaps and their optimal paths
        # for the weight w
        costmaps, optimal_paths, self.w, n = self.solve()
        return costmaps, optimal_paths, self.w, n

    def solve(self, begin=0):
        """ Returns a random vector as weights """
        # compute the learned costmaps and their optimal paths
        # for the weight w
        costmaps = []
        optimal_paths = []
        for _, i in enumerate(self.instances):
            costmap = get_costmap(i.phi, self.w)
            costmaps.append(costmap)
            i.learned_maps.append(costmap)
            _, _, paths = plan_paths(len(i.sample_trajectories), costmap,
                                     self.workspace, starts=i.sample_starts,
                                     targets=i.sample_targets)
            optimal_paths.append(paths)
            i.optimal_paths.append(paths)
        return costmaps, optimal_paths, self.w, 0
