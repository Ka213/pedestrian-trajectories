#!/usr/bin/env python

# Copyright (c) 2018, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                        Jim Mainprice on Sunday June 13 2018


from common_import import *

driectory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, driectory + os.sep + "..")
from my_utils.my_utils import *
from my_utils.output_costmap import *
from my_utils.environment import *
from nn.nn_utils import *
from pyrieef.geometry.workspace import *
from pyrieef.utils.misc import *
from math import *
from random import *
import optparse
from nn import my_dataset
from tqdm import tqdm
from numpy.testing import assert_allclose
import itertools


def samplerandpt(lims):
    """
        Sample a random point within limits
    """
    pt = np.array(lims.shape[0] * [0.])
    for j in range(pt.size):
        pt[j] = lims[j][0] + np.random.random() * (lims[j][1] - lims[j][0])
    return pt


def chomp_obstacle_cost(min_dist, epsilon):
    """
        Compute the cost function now (From CHOMP paper)
        If min_dist < 0, cost = -min_dist + epsilon/2
        If min_dist >= 0 && min_dist < epsilon, have a different cost
        If min_dist >= epsilon, cost = 0
    """
    cost = 0.
    if min_dist < 0:
        cost = - min_dist + 0.5 * epsilon
    elif min_dist <= epsilon:
        cost = (1. / (2 * epsilon)) * ((min_dist - epsilon) ** 2)
    return cost


def grids(workspace, grid_to_world, epsilon):
    """
        Creates a boolean matrix of occupancies
        To convert it to int or floats, use the following
        matrix.astype(int)
        matrix.astype(float)
    """
    # print "grid_to_world.shape : ", grid_to_world.shape
    m = grid_to_world.shape[0]
    assert grid_to_world.shape[1] == m

    costs = np.zeros((m, m))
    meshgrid = workspace.box.stacked_meshgrid(m)
    sdf = SignedDistanceWorkspaceMap(workspace)(meshgrid).T
    occupancy = sdf <= 0.
    test_grids = False
    if test_grids:
        # return [None, None, None]
        occupancy_tmp = np.zeros((m, m))
        sdf_tmp = np.zeros((m, m))
        for i, j in itertools.product(range(m), range(m)):
            [min_dist, obstacle_id] = workspace.min_dist(grid_to_world[i, j])
            sdf_tmp[i, j] = min_dist
            occupancy_tmp[i, j] = min_dist <= 0.
            costs[i, j] = chomp_obstacle_cost(min_dist, epsilon)

        assert_allclose(sdf_tmp, sdf)
        assert_allclose(occupancy_tmp, occupancy)
    else:
        for i, j in itertools.product(range(m), range(m)):
            costs[i, j] = chomp_obstacle_cost(sdf[i, j], epsilon)

    return [occupancy, sdf, costs]


def sample_circle_workspace(box,
                            nobjs_max=3,
                            random_max=False,
                            maxnumtries=100):
    """ Samples a workspace made of a maximum of
        nobjs_max circles that do not intersect
        todo replace the random environment script to use this function """
    workspace = Workspace(box)
    extent = box.extent()
    lims = np.array([[0., 1.], [0., 1.]])
    lims[0][1] = extent.x_max
    lims[0][0] = extent.x_min
    lims[1][1] = extent.y_max
    lims[1][0] = extent.y_min
    diagonal = box.diag()
    minrad = .10 * diagonal
    maxrad = .15 * diagonal
    nobj = nobjs_max if not random_max else int(ceil(random() * nobjs_max))
    for numtries in range(maxnumtries):
        r = minrad + random() * (maxrad - minrad)
        c = samplerandpt(lims)
        [min_dist, obstacle_id] = workspace.min_dist(c)
        if min_dist >= (r + .01 * diagonal):
            workspace.add_circle(c, r)
        if len(workspace.obstacles) >= nobj:
            return workspace
    return None


def random_environments(opt):
    lims = np.array([[0., 1.], [0., 1.]])
    size = np.array([opt.nb_points, opt.nb_points])
    numdatasets = opt.numdatasets
    nb_points = opt.nb_points
    nb_rbfs = opt.nb_rbfs
    sigma = opt.sigma
    nb_samples = opt.nb_samples
    if hasattr(opt, 'loss_stddev'):
        loss_stddev = opt.loss_stddev
    if hasattr(opt, 'loss_scalar'):
        loss_scalar = opt.loss_scalar
    if hasattr(opt, 'N'):
        N = opt.N

    save_workspace = True
    if opt.seed >= 0:
        print(("set random seed ({})".format(opt.seed)))
        np.random.seed(opt.seed)

    # Create a bunch of datasets
    datasets = [None] * numdatasets
    dataws = [None] * numdatasets

    print(("Num datasets : " + str(numdatasets)))
    for k in tqdm(list(range(numdatasets))):

        w, original_costmap, starts, targets, paths, centers = \
            create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples,
                                    Workspace())
        phi = get_phi(nb_points, centers, sigma, Workspace())

        algorithm = opt.filename.split('_')[:-1]
        map = np.zeros((nb_points, nb_points))
        if algorithm == 'learch':
            map = get_costmap(phi, np.exp(np.zeros((nb_rbfs ** 2))))
            map = get_learch_input(map, paths, starts, targets, loss_stddev,
                                   loss_scalar)
        elif algorithm == 'maxEnt':
            map = get_maxEnt_input(map, N, paths, targets, phi)
        elif algorithm == 'occ':
            map = get_occ_input(map, N, paths, targets)
        elif algorithm == 'loss_aug_occ':
            map = get_loss_aug_occ_input(map, paths, targets, loss_scalar,
                                         loss_stddev, N)
        elif algorithm == 'occ_learch':
            map = get_avg_learch_occ_input()
        # print(np.array([map, original_costmap]).dtype)
        datasets[k] = np.array([map, original_costmap])

        if save_workspace:  # TODO
            centers = np.asarray(centers)
            c1 = centers[:, 0]
            c2 = centers[:, 1]
            dataws[k] = phi  # np.array([w, c1, c2])

        if opt.display:
            show_multiple([map], [original_costmap], Workspace(), 'SHOW')
            break

    data = {}
    data["lims"] = lims
    data["size"] = size
    data["datasets"] = np.stack(datasets)

    print((np.stack(datasets).shape))
    print((np.stack(dataws).shape))
    workspaces = {}
    workspaces["lims"] = lims
    workspaces["size"] = size
    workspaces["datasets"] = np.stack(dataws)

    return data, workspaces


def get_dataset_id(data_id):
    options_data = my_dataset.get_yaml_options()
    options = dict_to_object(options_data[data_id])
    filename = options.filename + "." + options.type
    filepath = my_dataset.learning_data_dir() + os.sep + filename
    if os.path.exists(filepath) and os.path.isfile(filepath):
        data = my_dataset.CostmapDataset(filename)
        numtrain = data.train_inputs.shape[0]
        numtest = data.test_inputs.shape[0]
        numdatasets = numtrain + numtest
        assert options.numdatasets == numdatasets
        assert options.nb_points == data.train_targets.shape[1]
        assert options.nb_points == data.train_targets.shape[2]
        assert options.nb_points == data.train_inputs.shape[1]
        assert options.nb_points == data.train_inputs.shape[2]
        assert options.nb_points == data.test_targets.shape[1]
        assert options.nb_points == data.test_targets.shape[2]
        assert options.nb_points == data.test_inputs.shape[1]
        assert options.nb_points == data.test_inputs.shape[2]
        return data
    else:
        datasets, workspaces = random_environments(options)
        my_dataset.write_dictionary_to_file(datasets, filename)
        my_dataset.write_dictionary_to_file(
            workspaces, options.workspaces + "." + options.type)
        return get_dataset_id(data_id)


class RandomEnvironmentOptions:

    def __init__(self, dataset_id=None):
        if dataset_id is None:
            self._use_parser = True
        else:
            self._use_parser = False
            self._dataset_id = dataset_id

    def environment_parser(self):

        parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")

        parser.add_option('--dataset_id',
                          default="test_data", type="str",
                          dest='dataset_id',
                          help='Dataset ID')
        parser.add_option('--numdatasets',
                          default=1000, type="int", dest='numdatasets',
                          help='Number of datasets to generate')
        parser.add_option('--savefilename',
                          default='2dcostdata.t7', type="string",
                          dest='savefilename',
                          help='Filename to save results in\
                          (in local directory)')
        parser.add_option('--savematlabfile',
                          default=False, type="int",
                          dest='savematlabfile',
                          help='Save results in .mat format')
        parser.add_option('--nb_points',
                          default=28, type="int",
                          dest='nb_points',
                          help='Size of the squared grid (in pixels).\
                           X values go from 0-1')
        parser.add_option('--nb_rbfs',
                          default=4, type="int",
                          dest='nb_rbfs',
                          help='number of radial basis functions used')
        parser.add_option('--nb_samples',
                          default=5, type="int",
                          dest='nb_samples',
                          help='number of samples used for occupancy map')
        parser.add_option('--loss_scalar',
                          default=1, type="int",
                          dest='loss_scalar',
                          help='loss scalar used to compute loss augmentation')
        parser.add_option('--loss_stddev',
                          default=5, type="int",
                          dest='loss_stddev',
                          help='loss standard deviation used to compute loss '
                               'augmentation')
        parser.add_option('--N',
                          default=35, type="int",
                          dest='N',
                          help='number of iterations used to compute expected '
                               'state freuency')
        parser.add_option('--sigma',
                          default=0.1, type="float",
                          dest='sigma',
                          help='sigma of radial basis function')
        parser.add_option('--display',
                          default=True, type="int",
                          dest='display',
                          help='If set, displays the obstacle\
                           costs/occ grids in 2D')
        parser.add_option('--seed',
                          default=0, type="int",
                          dest='seed',
                          help='Random number seed. -ve values\
                           mean random seed')

        return parser

    def get_options(self):
        """ Load dataset options from file or option parser """
        if self._use_parser:
            parser = self.environment_parser()
            (options, args) = parser.parse_args()
            return options
        else:
            options = dataset.get_yaml_options()[self._dataset_id]
            return dict_to_object(options)


def remove_file_if_exists(file):
    if os.path.exists(file):
        os.remove(file)


if __name__ == '__main__':
    parser = RandomEnvironmentOptions()
    options = parser.get_options()
    dataset_paramerters = dict_to_object(
        my_dataset.get_yaml_options()[options.dataset_id])
    remove_file_if_exists("data/" + dataset_paramerters.filename + ".hdf5")
    remove_file_if_exists("data/" + dataset_paramerters.workspaces + ".hdf5")
    get_dataset_id(options.dataset_id)
