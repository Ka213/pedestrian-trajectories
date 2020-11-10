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

import h5py
import os
from pyrieef.utils import *
from pyrieef.utils.misc import *
import numpy as np
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import Trajectory
from tqdm import tqdm
from my_utils.environment import *
import time
from nn.nn_utils import *
from multiprocessing import Pool


# TODO write some import test


def learning_data_dir():
    path = home + '/../data/nn_datasets'
    return path


def write_data_to_file(data_out, filename='costdata2d_10k.hdf5'):
    directory = learning_data_dir()
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = h5py.File(directory + os.sep + filename, 'w')
    f.create_dataset("mydataset", data=data_out)
    f.close()


def load_data_from_file(filename='costdata2d_10k.hdf5'):
    with h5py.File(learning_data_dir() + os.sep + filename, 'r') as f:
        datasets = f['mydataset'][:]
    return datasets


def write_dictionary_to_file(data_out, filename='costdata2d_10k.hdf5'):
    directory = learning_data_dir()
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = h5py.File(directory + os.sep + filename, 'w')
    for key, value in list(data_out.items()):
        f.create_dataset(key, data=value)
    f.close()


def load_dictionary_from_file(filename='costdata2d_10k.hdf5'):
    datasets = {}
    with h5py.File(learning_data_dir() + os.sep + filename, 'r') as f:
        for d in f:
            # print("d : " + d)
            print(("f[d] : " + str(d)))
            datasets[str(d)] = f[d][:]
    return datasets


def load_data_from_hdf5(filename, train_per):
    """ Setup training / test data  """
    # Depracted
    print(('==> Loading dataset from: ' + filename))
    data = dict_to_object(load_dictionary_from_file(filename))
    print('==> Finished loading data')
    image_height = data.size[0]
    image_width = data.size[1]
    train_data = []
    test_data = []
    train_data_ids = []
    test_data_ids = []
    if train_data_ids and test_data_ids:
        print("We have some data ids")
        numTrain = len(train_data_ids)
        numTest = len(test_data_ids)
        numData = num_train + num_test
        for k in range(numTrain):
            train_data.append(data.datasets[train_data_ids[k]])
        for k in range(numTrain):
            test_data.append(data.datasets[test_data_ids[k]])
    else:
        # Load datasets afresh
        num_data = len(data.datasets)  # Total number of datasets
        num_train = int(round(train_per * num_data))
        num_test = num_data - num_train
        for k in range(num_data):
            if k < num_train:
                train_data.append(data.datasets[k])
                train_data_ids.append(k)
            else:
                test_data.append(data.datasets[k])
                test_data_ids.append(k)

        assert len(train_data) == num_train and len(test_data) == num_test

    print(('Num. total: {}, Num. train: {}; Num. test: {}'.format(
        num_data, num_train, num_test)))
    return train_data, test_data


def import_tf_data(filename='costdata2d_10k.hdf5'):
    """ Works with version 1.9.0rc1 """
    import tensorflow as tf
    rawdata = CostmapDataset(filename)
    # Assume that each row of
    # `inputs` corresponds to the same row as `targets`.
    assert rawdata.train_inputs.shape[0] == rawdata.train_targets.shape[0]
    dataset_train = tf.data.Dataset.from_tensor_slices((
        rawdata.train_inputs,
        rawdata.train_targets))
    print((dataset_train.output_types))
    print((dataset_train.output_shapes))
    dataset_test = None
    if rawdata.train_per < 1.:
        assert rawdata.test_inputs.shape[0] == rawdata.test_targets.shape[0]
        dataset_test = tf.data.Dataset.from_tensor_slices((
            rawdata.test_inputs,
            rawdata.test_targets))
    return dataset_train, dataset_test


def create_circles_workspace(box, ws):
    """ Creates circle dataset from array """
    workspace = Workspace(box)
    for i in range(ws[0].shape[0]):
        center = ws[0][i]
        radius = ws[1][i][0]
        if radius > 0:
            workspace.add_circle(center, radius)
    return workspace


def load_workspaces_from_file(filename='workspaces_1k_small.hdf5'):
    """ Load data from an hdf5 file """
    data_ws = dict_to_object(load_dictionary_from_file(filename))
    print((" -- size : ", data_ws.size))
    print((" -- lims : ", data_ws.lims.flatten()))
    print((" -- datasets.shape : ", data_ws.datasets.shape))
    print((" -- data_ws.shape : ", data_ws.datasets.shape))
    box = box_from_limits(
        data_ws.lims[0, 0], data_ws.lims[0, 1],
        data_ws.lims[1, 0], data_ws.lims[1, 1])
    workspaces = [None] * len(data_ws.datasets)
    for k, ws in enumerate(data_ws.datasets):
        workspaces[k] = create_circles_workspace(box, ws)
    return workspaces


def save_paths_to_file(paths, filename='paths_1k_demos.hdf5'):
    directory = learning_data_dir()
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = h5py.File(directory + os.sep + filename, 'w')
    for i, environment in enumerate(paths):
        grp = f.create_group(pad_zeros('environment_', i, len(paths)))
        for k, path in enumerate(environment):
            grp.create_dataset(
                pad_zeros('path_', k, len(environment)),
                data=np.array(path))


def load_paths_from_file(filename='paths_1k_demos.hdf5'):
    # np.set_printoptions(
    #    suppress=True,
    #    linewidth=200, precision=0,
    #    formatter={'float_kind': '{:8.0f}'.format})

    paths = []
    with h5py.File(learning_data_dir() + os.sep + filename, 'r') as f:
        for environment in f:
            # print(("f[d] : " + str(environment)))
            paths.append([])
            for path in f[environment]:
                p = f[environment][path][:]
                paths[-1].append(p)
                # print("path : " + str(path))
                # print(" --- : " + str(p.T))
    return paths


def save_trajectories_to_file(
        trajectories, filename='trajectories_1k_demos.hdf5'):
    nb_traj = len(trajectories)
    assert nb_traj > 0
    length = trajectories[0].x().size
    trajectories_data = [np.zeros(length)] * nb_traj
    for i, traj in enumerate(trajectories):
        trajectories_data[i] = traj.x()
    data = {}
    data["n"] = np.array([trajectories[0].n()])
    data["datasets"] = np.stack(trajectories_data)
    write_dictionary_to_file(data, filename=filename)


def load_trajectories_from_file(filename='trajectories_1k_small.hdf5'):
    """ Load data from an hdf5 file """
    data = dict_to_object(load_dictionary_from_file(filename))
    print((" -- trajectories * n : {}".format(data.n[0])))
    print((" -- trajectories * l : {}".format(len(data.datasets))))
    n = data.n[0]
    trajectories = [None] * len(data.datasets)
    for k, trj in enumerate(data.datasets):
        trajectories[k] = Trajectory(q_init=trj[:n], x=trj)
    return trajectories


def get_yaml_options():
    import yaml
    directory = learning_data_dir()
    filename = directory + os.sep + "synthetic_data_sdf.yaml"
    with open(filename, 'r') as stream:
        try:
            options_data = yaml.load(stream)
            # print(options_data)
        except yaml.YAMLError as exc:
            print(exc)
    return options_data


class CostmapDataset(object):

    def __init__(self, filename):
        print(('==> Loading dataset from: ' + filename))
        data = dict_to_object(load_dictionary_from_file(filename))
        self._max_index = 10000
        self._size_limit = False
        if not self._size_limit:
            self._max_index = len(data.datasets)
        self.train_per = 0.80
        print('Sorting out inputs and targets...')
        self.split_data(data)
        print((' - num. inputs : {}, shape : {}'.format(
            len(self.train_inputs),
            self.train_inputs.shape)))
        print((' - num. targets : {}, shape : {}'.format(
            len(self.train_targets),
            self.train_targets.shape)))

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(self.train_targets)

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def create_target(self, learning, workspace, costs, loss_stddev, loss_scalar, N,
                      workspace_box, i):
        print("costs: ", i)
        demonstrations = workspace[i].demonstrations
        starts = workspace[i].starts
        targets = workspace[i].targets
        costmap = costs[i]

        if learning == 'learch':
            map, op = get_learch_target(costmap, demonstrations, starts,
                                        targets, loss_stddev, loss_scalar,
                                        workspace_box)
        elif learning == 'maxEnt':
            map, f_exp, f_emp = get_maxEnt_target(costmap, N, demonstrations,
                                                  starts, targets,
                                                  workspace_box)
        elif learning == 'esf':
            map = get_esf_target(costmap, N, demonstrations, starts, targets,
                                 workspace_box)
        elif learning == 'loss_aug_esf':
            map = get_loss_aug_esf_target(costmap, demonstrations, starts,
                                          targets, loss_scalar, loss_stddev,
                                          N, workspace_box)
        elif learning == 'avg_esf_learch':
            map = get_avg_learch_esf_target()
        # pbar.update(1)
        return map.reshape(-1)

    def update_targets(self, costs, workspace, learning, loss_scalar,
                       loss_stddev, N):

        print("update target costmaps")
        box = EnvBox()
        lims = workspace[0].lims
        box.dim[0] = lims[0][1] - lims[0][0]
        box.dim[1] = lims[1][1] - lims[1][0]
        box.origin[0] = box.dim[0] / 2.
        box.origin[1] = box.dim[1] / 2.
        workspace_box = Workspace(box)
        time_0 = time.time()
        pbar = tqdm(total=self._max_index)
        if False:
            pool = Pool()
            x = range(len(self.train_targets))
            y = [(learning, workspace, costs, loss_stddev, loss_scalar, N,
                  workspace_box, i) for i in x]
            result = pool.starmap(self.create_target, y)
            self.train_targets = np.array(result)
            pbar.update(len(self.train_targets))
            x = range(len(self.test_targets))
            y = [(learning, workspace, costs, loss_stddev, loss_scalar, N,
                  workspace_box, i) for i in x]
            result = pool.starmap(self.create_target, y)
            self.test_targets = np.array(result)
            pool.close()
            pbar.update(len(self.test_targets))
        else:
            for i in range(len(self.train_targets)):
                self.train_targets[i] = self.create_target(learning, workspace,
                                                           costs, loss_stddev,
                                                           loss_scalar, N,
                                                           workspace_box,
                                                           i)
                pbar.update(1)
            for j in range(len(self.test_targets)):
                self.test_targets[j] = self.create_target(learning, workspace,
                                                          costs, loss_stddev,
                                                          loss_scalar, N,
                                                          workspace_box,
                                                          j + self._num_examples)
                pbar.update(1)
        pbar.close()
        print("took t : {} sec.".format(time.time() - time_0))
        self.normalize_targets()

    def reshape_data_to_tensors(self):

        def reshape_data_to_tensor(data):
            return data.reshape(data.shape[0], data.shape[1] * data.shape[2])

        self.train_inputs = reshape_data_to_tensor(self.train_inputs)
        self.train_targets = reshape_data_to_tensor(self.train_targets)
        self.test_inputs = reshape_data_to_tensor(self.test_inputs)
        self.test_targets = reshape_data_to_tensor(self.test_targets)

    def normalize_maps(self):
        """ normalize all maps"""

        self.normalize_targets()
        self.flip_inputs()

    def normalize_targets(self):
        def normalize(data):
            for costmap in data:
                costmap[:] = costmap - costmap.min()
                if costmap.max() != 0:
                    costmap[:] /= costmap.max()

        normalize(self.train_targets)
        normalize(self.test_targets)

    def flip_inputs(self):
        def flip(data):
            for costmap in data:
                costmap[:] = 1. - costmap

        flip(self.train_inputs)
        flip(self.test_inputs)

    def split_data(self, data):
        """ Load datasets afresh, train_per should be between 0 and 1 """
        assert self.train_per >= 0. and self.train_per < 1.
        print(" num_data : {}".format(len(data.datasets)))
        num_data = min(self._max_index, len(data.datasets))
        num_train = int(round(self.train_per * num_data))
        num_test = num_data - num_train
        print(" num_train : {}, num_test : {}".format(num_train, num_test))
        self.train_inputs = []
        self.train_targets = []
        self.test_inputs = []
        self.test_targets = []
        for i, d in enumerate(data.datasets):
            occupancy = d[0]
            costmap = d[2]
            if i < num_train:
                self.train_inputs.append(occupancy)
                self.train_targets.append(costmap)
            else:
                self.test_inputs.append(occupancy)
                self.test_targets.append(costmap)
            if i == self._max_index - 1 and self._size_limit:
                break
        self.train_inputs = np.array(self.train_inputs)
        self.train_targets = np.array(self.train_targets)
        if num_test > 0:
            self.test_inputs = np.array(self.test_inputs)
            self.test_targets = np.array(self.test_targets)
        assert len(self.train_inputs) == num_train
        assert len(self.test_inputs) == num_test

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._targets = self.train_targets[perm0]
            self._inputs = self.train_inputs[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            targets_rest_part = self._targets[start:self._num_examples]
            inputs_rest_part = self._inputs[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._targets = self._targets[perm]
                self._inputs = self._inputs[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            targets_new_part = self._targets[start:end]
            inputs_new_part = self._inputs[start:end]
            x = np.concatenate((inputs_rest_part, inputs_new_part), axis=0)
            y = np.concatenate((targets_rest_part, targets_new_part), axis=0)
            return x, y
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            x = self._inputs[start:end]
            y = self._targets[start:end]
            return x, y


class WorkspaceData:

    def __init__(self):
        self.workspace = None
        self.occupancy = None
        self.costmap = None
        self.signed_distance_field = None
        self.demonstrations = None
        self.starts = None
        self.targets = None
        self.lims = None


def load_workspace_dataset(basename="1k_small.hdf5"):
    file_ws = 'workspaces_' + basename
    file_cost = basename
    file_trj = 'trajectories_' + basename
    file_ep = 'endpoints_' + basename
    # dataset = WorkspaceData()
    workspaces = load_workspaces_from_file(file_ws)
    trajectories = load_paths_from_file(file_trj)
    data = dict_to_object(load_dictionary_from_file(file_cost))
    endpoints = load_data_from_file(file_ep)
    assert len(workspaces) == len(data.datasets)
    assert len(trajectories) == len(data.datasets)
    workspaces_dataset = [None] * len(workspaces)
    for k, (data_file, traj_file, ep_file) in enumerate(zip(data.datasets,
                                                            trajectories,
                                                            endpoints)):
        ws = WorkspaceData()
        ws.workspace = workspaces[k]
        ws.demonstrations = traj_file
        ws.starts = ep_file[0]
        ws.targets = ep_file[1]
        ws.occupancy = data_file[0]
        ws.signed_distance_field = data_file[1]
        ws.costmap = data_file[2]
        ws.lims = data.lims
        workspaces_dataset[k] = ws
    return workspaces_dataset
