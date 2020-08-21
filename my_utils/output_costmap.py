from common_import import *

from textwrap import wrap
import numpy as np
from matplotlib import cm
import matplotlib
import math
# print(matplotlib.get_backend())
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pyrieef.rendering.workspace_renderer as render


def show_example_trajectories(paths, pixel_map, starts, targets, viewer):
    """ Draw the example trajectories with their corresponding start and
        target state
    """
    assert len(starts) == len(targets) == len(paths)
    for s_w, t_w, path in zip(starts, targets, paths):
        trajectory = [None] * len(path)
        for i, p in enumerate(path):
            trajectory[i] = pixel_map.grid_to_world(np.array(p))
        c = cmap(np.random.rand())
        viewer.draw_ws_line_fill(trajectory, color=c)
        viewer.draw_ws_point(s_w, color=c)
        viewer.draw_ws_point(t_w, color=c)


def show_weights(viewer, weights, workspace):
    """ Draw an indication of where the weights have to increase and decrease
        draw a blue dot in the center of the weights which have to increase
        draw a green dot in the center of the weights which have to decrease
    """
    a = int(math.sqrt(len(weights)))
    centers = workspace.box.meshgrid_points(a)
    weights = weights.reshape((a, a)).T.reshape(len(weights))
    for i, (w_t, c) in enumerate(zip(weights, centers)):
        if w_t > 0:
            viewer.draw_ws_point(c, color='b', shape='x')
        else:
            viewer.draw_ws_point(c, color='g', shape='x')


def show_policy(pixel_map, predecessors, viewer):
    """ Draw the policy
        connect a state with its next state
        when following the current optimal policy
    """
    c = cmap(np.random.rand())
    for i, p in enumerate(predecessors):
        if p > 0:
            a = p % 40
            b = math.floor(p / 40)
            x = np.array((a, b))
            x = pixel_map.grid_to_world(x)
            a = i % 40
            b = math.floor(i / 40)
            y = np.array((a, b))
            y = pixel_map.grid_to_world(y)
            l = np.stack((x, y))
            viewer.draw_ws_line_fill(l, color=c)


def show_D(d, pixel_map, viewer):
    """ Draw an indication of D
        make a red cross on every point in D where D has to be decreased
        make a blue cross on every point in D where D has to be increased
    """
    d = d.T
    for i, d_row in enumerate(d):
        x_1, x_2, d_t = d_row
        s = pixel_map.grid_to_world(np.array((x_1, x_2)))
        if d_t < 0:
            viewer.draw_ws_point(s, color='r', shape='x')
        else:
            viewer.draw_ws_point(s, color='b', shape='x')


def show_optimal_paths(optimal_paths, pixel_map, viewer):
    """ Draw optimal trajectories """
    for path in optimal_paths:
        trajectory = [None] * len(path)
        for i, p in enumerate(path):
            trajectory[i] = pixel_map.grid_to_world(np.array(p))
        c = cmap(np.random.rand())
        viewer.draw_ws_line_fill(trajectory, color=c)


def show(costmap, workspace, show_result, starts=None, targets=None, paths=None,
         optimal_paths=None, weights=None, d=None, predecessors=None,
         title=None, directory=None):
    """ Show single map and optional optimal example trajectories,
        optimal trajectories, indication of weights, D or policy
    """
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
    viewer.draw_ws_img(costmap, interpolate="none")
    pixel_map = workspace.pixel_map(costmap.shape[0])

    if paths is not None:
        show_example_trajectories(paths, pixel_map, starts, targets, viewer)

    if optimal_paths is not None:
        show_optimal_paths(optimal_paths, pixel_map, viewer)

    if weights is not None:
        show_weights(viewer, weights, workspace)

    if d is not None:
        show_D(d, pixel_map, viewer)

    if predecessors is not None:
        show_policy(pixel_map, predecessors, viewer)

    viewer.remove_axis()

    if title is not None:
        viewer.set_title('\n'.join(wrap(title, 60)))

    if show_result == 'SHOW':
        viewer.show_once()
    elif show_result == 'SAVE':
        viewer.save_figure(directory)


def show_iteration(costmaps, original_costmap, workspace, show_result,
                   weights=None, starts=None, targets=None, paths=None,
                   optimal_paths=None, title=None, directory=None):
    """ Show multiple maps and optional optimal example trajectories,
        optimal trajectories, indication of weights, D or policy
    """
    pixel_map = workspace.pixel_map(costmaps[0].shape[0])
    rows = len(costmaps) + 1
    cols = len(paths)
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                    rows=rows, cols=cols,
                                    scale=rows / (rows * cols))

    viewer.draw_ws_img(original_costmap, interpolate="none")
    viewer._ax.set_title('Original Costmap', size=32 / cols)
    viewer.remove_axis()
    for j in range(cols):
        viewer.set_drawing_axis(j)
        viewer.remove_axis()

    for i in range(rows - 1):
        for j in range(cols):
            viewer.set_drawing_axis((i + 1) * cols + j)
            viewer.draw_ws_img(costmaps[i], interpolate="none")

            if weights is not None:
                show_weights(viewer, weights[i], workspace)

            if paths is not None:
                show_example_trajectories([paths[j]], pixel_map,
                                          [starts[j]], [targets[j]], viewer)
            if optimal_paths is not None:
                show_optimal_paths([optimal_paths[i][j]], pixel_map, viewer)
            viewer._ax.set_title('Learned Costmap: \n {}. Iteration \n'
                                 '{}. path'.format((i + 1), j + 1),
                                 size=32 / cols)
            viewer.remove_axis()
    viewer._fig.tight_layout()

    if title is not None:
        viewer.set_title('\n'.join(wrap(title, 60)))

    if show_result == 'SHOW':
        viewer.show_once()
    elif show_result == 'SAVE':
        viewer.save_figure(directory)


def show_multiple(costmaps, original_costmap, workspace, show_result,
                  weights=None, starts=None,
                  targets=None, paths=None, optimal_paths=None, step=1,
                  title=None, directory=None):
    """ Show multiple maps and optional optimal example trajectories,
        optimal trajectories, indication of weights, D or policy
    """
    pixel_map = workspace.pixel_map(original_costmap.shape[0])
    rows = math.ceil(math.sqrt(len(costmaps) / step + 1))
    cols = math.ceil((len(costmaps) / step + 1) / rows)
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                    rows=rows, cols=cols,
                                    scale=rows / (rows * cols))

    viewer.draw_ws_img(original_costmap, interpolate="none")
    viewer._ax.set_title('Original Costmap', size=32 / cols)
    viewer.remove_axis()

    for i in range(rows * cols - 1):
        viewer.set_drawing_axis(i + 1)
        if i < math.floor(len(costmaps) / step):
            viewer.draw_ws_img(costmaps[i * step], interpolate="none")
            viewer._ax.set_title('Learned Costmap: \n '
                                 '{}. '.format(i * step + 1), size=32 / cols)

            if weights is not None:
                show_weights(viewer, weights[i * step], workspace)
            if paths is not None:
                show_example_trajectories(paths, pixel_map, starts, targets,
                                          viewer)
            if optimal_paths is not None:
                show_optimal_paths(optimal_paths[i * step], pixel_map, viewer)

        viewer.remove_axis()
    viewer._fig.tight_layout()

    if title is not None:
        viewer.set_title('\n'.join(wrap(title, 60)), fontsize=32)

    if show_result == 'SHOW':
        viewer.show_once()
    elif show_result == 'SAVE':
        viewer.save_figure(directory)


def save_environment(filename, nb_points, nb_rbfs, sigma, nb_samples, w,
                     costmap, starts, targets, paths):
    """ Save the environment with given demonstrations in a file"""
    file = home + '/../data/environment/' + filename + '.npz'
    np.savez(file, nb_points=nb_points, nb_rbfs=nb_rbfs, sigma=sigma,
             nb_samples=nb_samples, w=w, costmap=costmap, starts=starts,
             targets=targets, paths=paths)
    return file


def load_environment(filename):
    """ Load an environment with demonstrations from a file """
    file = np.load(home + '/../data/environment/' + filename + '.npz',
                   allow_pickle=True)
    w = file['w']
    costmap = file['costmap']
    starts = file['starts']
    targets = file['targets']
    paths = file['paths']
    return w, costmap, starts, targets, paths


def load_environment_params(filename):
    """ Load an environment with demonstrations from a file"""
    file = np.load(home + '/../data/environment/' + filename + '.npz',
                   allow_pickle=True)
    nb_points = file['nb_points']
    nb_rbfs = file['nb_rbfs']
    sigma = file['sigma']
    nb_samples = file['nb_samples']
    return nb_points, nb_rbfs, sigma, nb_samples


def save_learch_params(directory, l):
    """ Save the hyperparametes of a LEARCH instance in a file """
    file = directory + '.npz'
    np.savez(file, loss_scalar=l._loss_scalar, loss_stddev=l._loss_stddev,
             learning_rate=l._learning_rate, stepsize_scalar=l._stepsize_scalar,
             l2_regularizer=l._l2_regularizer, proximal_regularizer=
             l._proximal_regularizer, exponentiatd_gd=l.exponentiated_gd)
    return file


def set_learch_params(directory, l):
    """ Set the hyperparameters of a LEARCH instance according to the values
        saved in the given file
    """
    file = np.load(directory)
    l._loss_scalar = file['loss_scalar']
    l._loss_stddev = file['loss_stddev']
    l._learing_rate = file['learning_rate']
    l._stepsize_scalar = file['stepsize_scalar']
    l._l2_regularizer = file['l2_regularizer']
    l._proximal_regularizer = file['proximal_regularizer']
    l.exponentiated_gd = file['exponentiated_gd']
    return l


def save_maxEnt_params(directory, m):
    """ Save the hyperparametes of a maxEnt instance in a file """
    file = directory + '.npz'
    np.savez(file, learning_rate=m._learning_rate, stepsize_scalar=
    m._stepsize_scalar, N=m._N)
    return file


def set_maxEnt_params(directory, m):
    """ Set the hyperparameters of a maxEnt instance according to the values
        saved in the given file
    """
    file = np.load(directory)
    m._learning_rate = file['learning_rate']
    m._stepsize_scalar = file['stepsize_scalar']
    m._N = file['N']
    return m


cmap = plt.get_cmap('viridis')
