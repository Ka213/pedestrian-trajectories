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
import common_import

from textwrap import wrap
import matplotlib.pyplot as plt
import pyrieef.rendering.workspace_renderer as render
from pyrieef.geometry.interpolation import *
from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *
from pyrieef.learning.inverse_optimal_control import *
from learch2.learch import *


def show_map(costmap, workspace, starts=None, targets=None, paths=None,
             title=None, directory=None):
    """ Show single map (with example trajectories) """
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
    viewer.draw_ws_img(costmap, interpolate="none")
    # Show example trajectories
    if paths is not None:
        assert len(starts) == len(targets) == len(paths)
        pixel_map = workspace.pixel_map(costmap.shape[0])
        for s_w, t_w, path in zip(starts, targets, paths):
            trajectory = [None] * len(path)
            for i, p in enumerate(path):
                trajectory[i] = pixel_map.grid_to_world(np.array(p))
            c = cmap(np.random.rand())
            viewer.draw_ws_line(trajectory, color=c)
            viewer.draw_ws_point(s_w, color=c)
            viewer.draw_ws_point(t_w, color=c)
    viewer.remove_axis()
    if title is not None:
        viewer.set_title(title)
    if show_result:
        viewer.show_once()
    else:
        viewer.save_figure(directory)


def show_weight(costmap, w, workspace, starts=None, targets=None, paths=None,
                optimal_paths=None, title=None, directory=None):
    """ Show single map (with example trajectories)
        and indication of weight changes
        If the weight has to increase,
        draw blue cross in the center of the corresponding rbf
        If the weight has to decrease,
        draw green dot in the center of the corresponding rbf
    """
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
    viewer.draw_ws_img(costmap, interpolate="none")
    # Show example trajectories
    if paths is not None:
        assert len(starts) == len(targets) == len(paths)
        pixel_map = workspace.pixel_map(costmap.shape[0])
        for s_w, t_w, path, o_path in zip(starts, targets, paths,
                                          optimal_paths[-1]):
            trajectory = [None] * len(path)
            op_trajectory = [None] * len(o_path)
            for i, p in enumerate(path):
                trajectory[i] = pixel_map.grid_to_world(np.array(p))
            for i, p in enumerate(o_path):
                op_trajectory[i] = pixel_map.grid_to_world(np.array(p))
            c = cmap(0.1)
            viewer.draw_ws_line(trajectory, color=c)
            viewer.draw_ws_point(s_w, color=c)
            viewer.draw_ws_point(t_w, color=c)
            c = cmap(0.8)
            viewer.draw_ws_line(op_trajectory, color=c)
    viewer.remove_axis()
    centers = workspace.box.meshgrid_points(5)
    w = w.reshape((5, 5)).T.reshape(25)
    for i, (w_t, c) in enumerate(zip(w, centers)):
        if w_t > 0:
            viewer.draw_ws_point(c, color='b', shape='x')
        else:
            viewer.draw_ws_point(c, color='g', shape='o')
    if title is not None:
        viewer.set_title(title)
    if show_result:
        viewer.show_once()
    else:
        viewer.save_figure(directory)


def show_multiple_weights(costmaps, weights, workspace, starts=None,
                          targets=None, paths=None, optimal_paths=None,
                          title=None, directory=None):
    """ Show multiple maps (with example trajectories)
        and indication of weight changes
        If example trajectories are shown
        one row corresponds to one iteration of LEARCH
        and one column to the example trajectory and optimal trajectory
        If the weight has to increase,
        draw blue cross in the center of the corresponding rbf
        If the weight has to decrease,
        draw green dot in the center of the corresponding rbf
    """
    centers = workspace.box.meshgrid_points(5)
    if paths is not None:
        assert len(starts) == len(targets) == len(paths)
        r = len(costmaps)
        co = len(paths)
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                        rows=r, cols=co, scale=2 * (1 / r * co))
        for i, (costmap, w) in enumerate(zip(costmaps, weights)):
            pixel_map = workspace.pixel_map(costmap.shape[0])
            w = w.reshape((5, 5)).T.reshape(25)
            for k, (s_w, t_w, path, optimal_path) in \
                    enumerate(zip(starts, targets, paths, optimal_paths[i])):
                trajectory = [None] * len(path)
                optimal_trajectory = [None] * len(optimal_path)
                viewer.set_drawing_axis(i * co + k)
                viewer.remove_axis()
                viewer.draw_ws_img(costmap, interpolate="none")
                for l, p in enumerate(path):
                    trajectory[l] = pixel_map.grid_to_world(np.array(p))
                for l, op in enumerate(optimal_path):
                    optimal_trajectory[l] = pixel_map. \
                        grid_to_world(np.array(op))
                c = cmap(0.2)
                viewer.draw_ws_line_fill(trajectory, color=c)
                viewer.draw_ws_point(s_w, color=c)
                viewer.draw_ws_point(t_w, color=c)
                c = cmap(0.8)
                viewer.draw_ws_line_fill(optimal_trajectory, color=c)
                for w_t, c in zip(w, centers):
                    if w_t > 0:
                        viewer.draw_ws_point(c, color='b', shape='x')
                    else:
                        viewer.draw_ws_point(c, color='g', shape='o')
    else:
        r = math.ceil(math.sqrt(len(costmaps)))
        c = math.ceil(len(costmaps) / r)
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                        rows=r, cols=c,
                                        scale=(1 / (r * c)) * 10)
        for i, (costmap, w) in enumerate(zip(costmaps, weights)):
            viewer.set_drawing_axis(i)
            viewer.remove_axis()
            viewer.draw_ws_img(costmap, interpolate="none")
            w = w.reshape((5, 5)).T.reshape(25)
            for w_t, c in zip(w, centers):
                if w_t > 0:
                    viewer.draw_ws_point(c, color='b', shape='x')
                else:
                    viewer.draw_ws_point(c, color='g', shape='o')
    if title is not None:
        viewer.set_title(title)
    if show_result:
        viewer.show_once()
    else:
        viewer.save_figure(directory)


def show_multiple_maps(costmaps, workspace, starts=None, targets=None, paths=None,
                       optimal_paths=None, title=None, directory=None):
    """ Show multiple maps (with example trajectories) in one plot
        If example trajectories are shown
        one row corresponds to one iteration of LEARCH
        and one column to the example trajectory and optimal trajectory
    """
    if paths is not None:
        assert len(starts) == len(targets) == len(paths)
        r = len(costmaps)
        co = len(paths)
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                        rows=r, cols=co, scale=(1 / r * co))
        for i, costmap in enumerate(costmaps):
            pixel_map = workspace.pixel_map(costmap.shape[0])
            for k, (s_w, t_w, path, optimal_path) in \
                    enumerate(zip(starts, targets, paths, optimal_paths[i])):
                trajectory = [None] * len(path)
                optimal_trajectory = [None] * len(optimal_path)
                viewer.set_drawing_axis(i * co + k)
                viewer.remove_axis()
                viewer.draw_ws_img(costmap, interpolate="none")
                for l, p in enumerate(path):
                    trajectory[l] = pixel_map.grid_to_world(np.array(p))
                for l, op in enumerate(optimal_path):
                    optimal_trajectory[l] = pixel_map. \
                        grid_to_world(np.array(op))
                c = cmap(np.random.rand())
                viewer.draw_ws_line_fill(trajectory, color=c)
                viewer.draw_ws_line_fill(optimal_trajectory, color=c)
                viewer.draw_ws_point(s_w, color=c)
                viewer.draw_ws_point(t_w, color=c)
    else:
        r = math.ceil(math.sqrt(len(costmaps)))
        c = math.ceil(len(costmaps) / r)
        viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                        rows=r, cols=c,
                                        scale=(1 / (r * c)) * 10)
        for i, costmap in enumerate(costmaps):
            viewer.set_drawing_axis(i)
            viewer.remove_axis()
            viewer.draw_ws_img(costmap, interpolate="none")
    if title is not None:
        viewer.set_title(title)
    if show_result:
        viewer.show_once()
    else:
        viewer.save_figure(directory)

def show_D(costmap, d, workspace,
     starts, targets, paths, optimal_paths, directory=None):
    """ Show a map with the indication where the costs have to be decreased along
        the example and optimal trajectories
    """
    assert len(starts) == len(targets) == len(paths)
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True, cols= 2)
    pixel_map = workspace.pixel_map(costmap.shape[0])
    # Example paths where cost have to be decreased
    viewer.draw_ws_img(costmap, interpolate="none")
    for k, (s_w, t_w, path) in enumerate(zip(starts, targets, paths)):
        trajectory = [None] * len(path)
        for l, p in enumerate(path):
            trajectory[l] = pixel_map.grid_to_world(np.array(p))
        c = cmap(0.8)
        viewer.draw_ws_line_fill(trajectory, color=c)
        viewer.draw_ws_point(s_w, color=c)
        viewer.draw_ws_point(t_w, color=c)
    viewer.set_drawing_axis(1)
    # Optimal paths where cost have to be decreased
    viewer.draw_ws_img(costmap, interpolate="none")
    for k, (s_w, t_w, path) in enumerate(zip(starts, targets, optimal_paths)):
        trajectory = [None] * len(path)
        for l, p in enumerate(path):
            trajectory[l] = pixel_map.grid_to_world(np.array(p))
        c = cmap(0.2)
        viewer.draw_ws_line_fill(trajectory, color=c)
        viewer.draw_ws_point(s_w, color=c)
        viewer.draw_ws_point(t_w, color=c)
    # Show indication of D
    d = d.T
    for i, d_row in enumerate(d):
        x_1, x_2, d_t = d_row
        s = pixel_map.grid_to_world(np.array((x_1, x_2)))
        if d_t > 0:
            viewer.set_drawing_axis(1)
            viewer.draw_ws_point(s, color='b', shape='x')
        else:
            viewer.set_drawing_axis(0)
            viewer.draw_ws_point(s, color='g', shape='x')
        viewer.remove_axis()
    if show_result:
        viewer.show_once()
    else:
        viewer.save_figure(directory)


def plot_error_avg(error, nb_samples, nb_runs, directory):
    """ Plot error over different number of samples, average over seeds """
    x = np.arange(nb_samples) + 1
    y = np.average(error, axis=0)
    plt.figure()
    plt.plot(x, y)
    plt.ylabel('error')
    plt.xlabel('# of samples')
    plt.xticks(x)
    y_stddev = np.std(error, axis=1)
    plt.errorbar(x, y, yerr=y_stddev, capsize=2)
    plt.title(
        '\n'.join(wrap('error over different number of samples averaged '
                       'over {} different environments'.format(nb_runs), 60)))
    plt.savefig(directory)


def plot_error_fix_env(error, nb_samples, directory):
    """ Plot error over different number of samples for fixed seed """
    x = np.arange(nb_samples) + 1
    plt.figure()
    plt.plot(x, error)
    plt.ylabel('error')
    plt.xlabel('# of samples')
    plt.xticks(x)
    plt.title('error over different number of samples over one environment')
    plt.savefig(directory)


def plot_error_fix_nbsamples(error, nb_samples, nb_runs, directory):
    """ Plot error over different seeds for fixed number of samples """
    x = np.arange(nb_runs) + 1
    plt.figure()
    plt.plot(x, error)
    plt.ylabel('error')
    plt.xlabel('runs')
    plt.xticks(x)
    plt.title('error over different seeds for {} samples'.format(nb_samples))
    plt.savefig(directory)


show_result = False
cmap = plt.get_cmap('viridis')
