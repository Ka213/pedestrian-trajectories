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


def save_map(costmap, workspace, directory,
             starts=None, targets=None, paths=None, title=None):
    """ Save plot of single map (with example trajectories) """
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
    viewer.draw_ws_img(costmap, interpolate="none")
    viewer.remove_axis()
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
    if title is not None:
        viewer.set_title(title)
    viewer.save_figure(directory)


def show_map(costmap, workspace,
             starts=None, targets=None, paths=None, title=None):
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
    viewer.show_once()


def show_multiple_maps(costmaps, workspace, starts=None, targets=None,
                       paths=None, optimal_paths=None, title=None, scale=20):
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
    viewer.show_once()


def plot_error_avg(error, nb_samples, nb_runs):
    """ Plot error over different number of samples, average over seeds """
    x = np.arange(nb_samples) + 1
    plt.figure()
    plt.plot(x, error)
    plt.ylabel('error')
    plt.xlabel('# of samples')
    plt.xticks(x)
    plt.title(
        '\n'.join(wrap('error over different number of samples averaged '
                       'over %i different environments' % nb_runs, 60)))
    plt.savefig('../figures/diff_nb_samples_avg_seeds.png')


def plot_error_fix_env(error, nb_samples):
    """ Plot error over different number of samples for fixed seed """
    x = np.arange(nb_samples) + 1
    plt.figure()
    plt.plot(x, error)
    plt.ylabel('error')
    plt.xlabel('# of samples')
    plt.xticks(x)
    plt.title('error over different number of samples over one environment')
    plt.savefig('../figures/diff_nb_samples.png')


def plot_error_fix_nbsamples(error, nb_samples, nb_runs):
    """ Plot error over different seeds for fixed number of samples """
    x = np.arange(nb_runs) + 1
    plt.figure()
    plt.plot(x, error)
    plt.ylabel('error')
    plt.xlabel('runs')
    plt.xticks(x)
    plt.title('error over different seeds for %i samples' % nb_samples)
    plt.savefig('../figures/diff_seeds.png')


cmap = plt.get_cmap('viridis')
