from common_import import *

from textwrap import wrap
import numpy as np
from matplotlib import cm
import matplotlib as mlt
import matplotlib.animation as animation
import math
import subprocess
# print(matplotlib.get_backend())
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import pyrieef.rendering.workspace_renderer as render


def show_paths(paths, pixel_map, starts, targets, viewer):
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
        viewer.draw_ws_point(trajectory[0], color=c)
        viewer.draw_ws_point(trajectory[-1], color=c)


def show_weights(viewer, weights, centers):
    """ Draw an indication of where the weights have to increase and decrease
        draw a blue dot in the center of the weights which have to increase
        draw a green dot in the center of the weights which have to decrease
    """
    for i, (w_t, c) in enumerate(zip(weights, centers)):
        if w_t > 0:
            viewer.draw_ws_point([c[1], c[0]], color='b', shape='x')
        else:
            viewer.draw_ws_point([c[1], c[0]], color='g', shape='x')


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


def show(costmap, workspace, show_result, starts=None, targets=None, paths=None,
         ex_paths=None, weights=None, d=None, predecessors=None,
         title=None, directory=None, centers=None):
    """ Show a single map with demonstrations and example paths
    """
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True)
    viewer.draw_ws_img(costmap, interpolate="none")
    viewer._colorbar.ax.tick_params(labelsize=FONTSIZE)
    pixel_map = workspace.pixel_map(costmap.shape[0])

    if paths is not None:
        show_paths(paths, pixel_map, starts, targets, viewer)

    if ex_paths is not None:
        show_paths(ex_paths, pixel_map, starts, targets, viewer)

    if weights is not None:
        show_weights(viewer, weights, centers)

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


def show_iteration(costmaps, original_costmaps, workspace, show_result,
                   weights=None, starts=None, targets=None, paths=None,
                   centers=None, ex_paths=None, title=None, directory=None):
    """ Show multiple maps with demonstrations and example paths
        One row for each iteration
        One column for each demonstrations with its corresponding example path
    """
    pixel_map = workspace.pixel_map(costmaps[0].shape[0])
    rows = len(costmaps) + 1
    cols = max(len(paths), len(original_costmaps))
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                    rows=rows, cols=cols,
                                    scale=rows / (rows * cols))

    for i in range(cols):
        viewer.set_drawing_axis(i)
        if i < len(original_costmaps):
            viewer.draw_ws_img(original_costmaps[i], interpolate="none")
            viewer._ax.set_title('Training Costmap', size=32 / cols)
        viewer.remove_axis()

    for i in range(rows - 1):
        for j in range(cols):
            viewer.set_drawing_axis((i + 1) * cols + j)
            viewer.draw_ws_img(costmaps[i], interpolate="none")

            if weights is not None:
                show_weights(viewer, weights[i], centers)

            if paths is not None:
                print(len(paths))
                print(len(starts))
                show_paths([paths[j]], pixel_map, [starts[j]], [targets[j]],
                           viewer)
            if ex_paths is not None:
                show_paths([ex_paths[i][j]], pixel_map, [starts[j]],
                           [targets[j]], viewer)
            viewer._ax.set_title('Learned Costmap: \n {}. Iteration \n'
                                 '{}. path'.format((i + 1), j + 1),
                                 size=32 / cols)
            viewer.remove_axis()
    viewer._fig.tight_layout()

    if title is not None:
        viewer._fig.suptitle(title, fontsize=15)

    if show_result == 'SHOW':
        viewer.show_once()
    elif show_result == 'SAVE':
        viewer.save_figure(directory)


def show_multiple(costmaps, original_costmaps, workspace, show_result,
                  weights=None, centers=None, starts=None,
                  targets=None, paths=None, ex_paths=None,
                  title=None, directory=None):
    """ Show multiple maps with demonstrations and example paths
        plot all demonstrations and example paths of one iteration in one map
    """
    pixel_map = workspace.pixel_map(original_costmaps[0].shape[0])
    cols = max(math.ceil(math.sqrt(len(costmaps))), len(original_costmaps) + 1)
    rows = math.ceil(len(costmaps) / cols) + 1
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                    rows=rows, cols=cols,
                                    scale=rows / (rows * cols))
    for i in range(cols):
        viewer.set_drawing_axis(i)
        if i < len(original_costmaps):
            viewer.draw_ws_img(original_costmaps[i], interpolate="none")
            # viewer._ax.set_title('Training Costmap', size=32 / cols)
        viewer.remove_axis()

    for i in range(cols, rows * cols):
        viewer.set_drawing_axis(i)
        if i < len(costmaps) + cols:
            viewer.draw_ws_img(costmaps[i - cols], interpolate="none")
            # viewer._ax.set_title('Learned Costmap: \n {}.'.format(i + 1 - cols),
            #                     size=32 / cols)
            if weights is not None:
                show_weights(viewer, weights[i - cols], centers)
            if paths is not None:
                show_paths(paths, pixel_map, starts, targets, viewer)
            if ex_paths is not None:
                show_paths(ex_paths[i - cols], pixel_map, starts, targets,
                           viewer)
        viewer.remove_axis()

    # viewer._colorbar.ax.tick_params(labelsize=FONTSIZE)
    viewer._fig.tight_layout()

    if title is not None:
        viewer.set_title('\n'.join(wrap(title, 60)), fontsize=32)
    if show_result == 'SHOW':
        viewer.show_once()
    elif show_result == 'SAVE':
        viewer.save_figure(directory)


def show_esf_maps(costmaps, original_costmap, workspace, show_result, esf,
                  loss_aug_maps, loss_aug_esf, weights=None, centers=None,
                  starts=None, targets=None, paths=None, ex_paths=None,
                  title=None, directory=None):
    """ show the costmap with its log augmentation as well as the corresponding
        expected state frequencies
    """
    pixel_map = workspace.pixel_map(original_costmap.shape[0])
    cols = 4
    rows = len(costmaps) + 1
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                    rows=rows, cols=cols,
                                    scale=rows / (rows * cols))
    viewer.draw_ws_img(original_costmap, interpolate="none")
    viewer._ax.set_title('Training Costmap', size=32 / cols)
    for i in range(cols):
        viewer.set_drawing_axis(i)
        viewer.remove_axis()

    for i in range(1, rows):
        viewer.set_drawing_axis(i * cols)
        viewer.remove_axis()
        viewer.draw_ws_img(costmaps[i - 1], interpolate="none")
        viewer._ax.set_title('learned costmap: \n {}.'.format(i),
                             size=32 / cols)
        if weights is not None and centers is not None:
            show_weights(viewer, weights[i - 1], centers)
        viewer.set_drawing_axis(i * cols + 1)
        viewer.remove_axis()
        viewer.draw_ws_img(loss_aug_maps[i - 1], interpolate="none")
        viewer._ax.set_title('loss augmented costmap: \n {}.'.format(i),
                             size=32 / cols)
        viewer.set_drawing_axis(i * cols + 2)
        viewer.remove_axis()
        viewer.draw_ws_img(esf[i - 1], interpolate="none")
        viewer._ax.set_title('occupancy: \n {}.'.format(i), size=32 / cols)
        if paths is not None:
            show_paths(paths, pixel_map, starts, targets, viewer)
        if ex_paths is not None:
            show_paths(ex_paths[i - 1], pixel_map, starts, targets, viewer)
        if weights is not None and centers is not None:
            show_weights(viewer, weights[i - 1], centers)
        viewer.set_drawing_axis(i * cols + 3)
        viewer.remove_axis()
        viewer.draw_ws_img(loss_aug_esf[i - 1], interpolate="none")
        viewer._ax.set_title('loss augmented occupancy: \n {}.'.format(i),
                             size=32 / cols)
        if paths is not None:
            show_paths(paths, pixel_map, starts, targets, viewer)
        if ex_paths is not None:
            show_paths(ex_paths[i - 1], pixel_map, starts, targets, viewer)
    viewer._fig.tight_layout()

    if title is not None:
        viewer.set_title('\n'.join(wrap(title, 60)), fontsize=32)

    if show_result == 'SHOW':
        viewer.show_once()
    elif show_result == 'SAVE':
        viewer.save_figure(directory)


def show_predictions(costmap, original_costmap, workspace, show_result,
                     starts=None, targets=None, paths=None, ex_paths=None,
                     title=None, directory=None):
    """ Show multiple maps with demonstrations and example paths """
    pixel_map = workspace.pixel_map(original_costmap.shape[0])
    cols = min(5, len(ex_paths))
    rows = math.ceil(len(ex_paths) / 5) + 1
    viewer = render.WorkspaceDrawer(workspace, wait_for_keyboard=True,
                                    rows=rows, cols=cols,
                                    scale=rows / (rows * cols))
    viewer.draw_ws_img(original_costmap, interpolate="none")
    # viewer._ax.set_title('Training Costmap', size=32 / cols)
    for i in range(cols):
        viewer.set_drawing_axis(i)
        viewer.remove_axis()

    for i in range(cols, rows * cols):
        viewer.set_drawing_axis(i)
        if i < len(ex_paths) + cols:
            viewer.draw_ws_img(costmap, interpolate="none")

            if paths is not None:
                show_paths([paths[i - cols]], pixel_map, [starts[i - cols]],
                           [targets[i - cols]], viewer)
            if ex_paths is not None:
                show_paths([ex_paths[i - cols]], pixel_map,
                           [starts[i - cols]], [targets[i - cols]], viewer)
        viewer.remove_axis()
    viewer._fig.tight_layout()

    if title is not None:
        viewer.set_title('\n'.join(wrap(title, 60)), fontsize=32)
    #viewer._colorbar.ax.tick_params(labelsize=FONTSIZE)
    if show_result == 'SHOW':
        viewer.show_once()
    elif show_result == 'SAVE':
        viewer.save_figure(directory + '.pdf')


def show_3D(costmap, workspace, show_result, starts=None, targets=None,
            paths=None, ex_paths=None, directory=None, centers=None):
    """ show costmap with demonstrations and example paths in 3D """
    pixel_map = workspace.pixel_map(costmap.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(costmap.shape[0])
    x_1, x_2 = np.meshgrid(x, x)
    surf = ax.plot_surface(x_1, x_2, costmap.T, cmap=cm.magma, shade=False,
                           alpha=0.7)
    if paths is not None:
        for s, t, p in zip(starts, targets, paths):
            c = cmap(np.random.rand())
            x = np.asarray(p)[:, 0]
            y = np.asarray(p)[:, 1]
            path = ax.plot3D(x, y, costmap[x, y], color=c, alpha=1)
            s = pixel_map.world_to_grid(s)
            t = pixel_map.world_to_grid(t)
            p = np.array(p)
            start = ax.scatter3D(p[0, 0], p[0, 1], costmap[p[0, 0], p[0, 1]],
                                 color=c)
            target = ax.scatter3D(p[-1, 0], p[-1, 1], costmap[p[-1, 0],
                                                              p[-1, 1]], color=c)
    if ex_paths is not None:
        for p in zip(ex_paths):
            c = cmap(np.random.rand())
            x = np.asarray(p)[:, 0]
            y = np.asarray(p)[:, 1]
            path = ax.plot3D(x, y, costmap[x, y], color=c)
    if centers is not None:
        for c in zip(centers):
            color = 'b'
            c = pixel_map.world_to_grid(c[0])
            ax.scatter3D(c[1], c[0], costmap[c[1], c[0]], color=color,
                         marker='x')
    plt.tick_params(labelsize=15)
    if show_result == 'SHOW':
        plt.show()
    elif show_result == 'SAVE':
        plt.savefig(directory)


def show_multiple_3D(costmaps, workspace, show_result, labels=None, starts=None,
                     targets=None, paths=None, ex_paths=None, directory=None):
    """ show multiple costmaps in one plot
        with demonstrations and example paths
    """
    pixel_map = workspace.pixel_map(costmaps[0].shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(costmaps[0].shape[0])
    x_1, x_2 = np.meshgrid(x, x)
    plots = []
    for costmap in zip(costmaps):
        c = cmap(np.random.rand())
        costmap = np.array(costmap)[0]
        surf = ax.plot_surface(x_1, x_2, costmap.T, color=c,  # cmap=cm.magma,
                               shade=False, alpha=0.5)
        proxy = mlt.lines.Line2D([0], [0], linestyle="none", c=c,
                                 marker='o')
        plots.append(proxy)
    ax.legend(plots, labels, numpoints=1)
    if paths is not None:
        for s, t, p in zip(starts, targets, paths):
            c = cmap(np.random.rand())
            x = np.asarray(p)[:, 0]
            y = np.asarray(p)[:, 1]
            path = ax.plot3D(x, y, costmap[x, y], color=c)
            s = pixel_map.world_to_grid(s)
            t = pixel_map.world_to_grid(t)
            start = ax.scatter3D(s[0], s[1], costmap[s[0], s[1]], color=c)
            target = ax.scatter3D(t[0], t[1], costmap[t[0], t[1]], color=c)
    if ex_paths is not None:
        for op in zip(ex_paths):
            c = cmap(np.random.rand())
            x = np.asarray(op)[:, 0]
            y = np.asarray(op)[:, 1]
            path = ax.plot3D(x, y, costmap[x, y], color=c)

    if show_result == 'SHOW':
        plt.show()
    elif show_result == 'SAVE':
        plt.savefig(directory)


def animated_plot(maps, workspace, show_result, starts=None, targets=None,
                  paths=None, ex_paths=None, directory=None):
    """ show the given costmaps sequentially in one loop """

    def update_plot(frame_number, maps, s, t, p, op, surface, plot_paths):
        surface[0].remove()
        surface[0] = ax.plot_surface(x_1, x_2, maps[frame_number].T, cmap="magma",
                                     alpha=0.6)
        k = 0
        if len(p) > 0:
            for i, (path_u, start_u, target_u) in enumerate(zip(p, s, t)):
                plot_paths[i * 3].set_data(path_u[:, 0], path_u[:, 1])
                plot_paths[i * 3].set_3d_properties(
                    maps[frame_number][path_u[:, 0], path_u[:, 1]])
                plot_paths[i * 3 + 1].set_data([start_u[0]], [start_u[1]])
                plot_paths[i * 3 + 1].set_3d_properties(
                    [maps[frame_number][start_u[0], start_u[1]]])
                plot_paths[i * 3 + 2].set_data([target_u[0]], [target_u[1]])
                plot_paths[i * 3 + 2].set_3d_properties(
                    [maps[frame_number][target_u[0], target_u[1]]])
                k = i * 3 + 2 + 1

        if len(op) > 0:
            for j, path_u in enumerate(op[frame_number]):
                x = np.asarray(path_u)[:, 0]
                y = np.asarray(path_u)[:, 1]
                plot_paths[j + k].set_data(x, y)
                plot_paths[j + k].set_3d_properties(maps[frame_number][x, y])
        return surface, plot_paths

    fps = 5  # frame per sec
    pixel_map = workspace.pixel_map(maps[0].shape[0])
    x = np.arange(maps[0].shape[0])
    x_1, x_2 = np.meshgrid(x, x)
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    maps = np.asarray(maps)
    surface = [ax.plot_surface(x_1, x_2, maps[0].T, cmap=cm.magma, shade=False,
                               alpha=0.6)]
    ax.set_zlim(max(0, np.amin(maps)), np.amax(maps))

    c = cmap(np.random.rand())
    s = []
    t = []
    p = []
    plot_paths = []
    if paths is not None:
        for i, (start, target, path) in enumerate(zip(starts, targets, paths)):
            start = pixel_map.world_to_grid(start)
            start = start.astype(int)
            s.append(start)
            target = pixel_map.world_to_grid(target)
            t.append(target)
            path = np.asarray(path)
            p.append(path)
            plot_paths.append(ax.plot(path[:, 0], path[:, 1],
                                      maps[0][path[:, 0], path[:, 1]], color=c)[0])
            plot_paths.append(ax.plot([start[0]], [start[1]],
                                      [maps[0][start[0], start[1]]], marker='o',
                                      color=c)[0])
            plot_paths.append(ax.plot([target[0]], [target[1]],
                                      [maps[0][target[0], target[1]]], marker='o',
                                      color=c)[0])

    p = []
    if ex_paths is not None:
        for ex_path in ex_paths[0]:
            ex_path = np.asarray(ex_path)
            plot_paths.append(ax.plot(ex_path[:, 0], ex_path[:, 1],
                                      maps[0][ex_path[:, 0], ex_path[:, 1]],
                                      color=c)[0])
        p = np.asarray(ex_paths)

    ani = animation.FuncAnimation(fig, update_plot, maps.shape[0],
                                  fargs=(maps, s, t, p, p, surface, plot_paths),
                                  interval=1000 / fps)
    if show_result == 'SHOW':
        plt.show()
    elif show_result == 'SAVE':
        ani.save(directory + '.mp4', writer='ffmpeg', fps=fps)
        ani.save(directory + '.gif', writer='imagemagick', fps=fps)
        # reduce the size of the GIF file
        cmd = 'magick convert {}.gif -fuzz 5%% -layers Optimize {}_r.gif' \
            .format(directory, directory)
        subprocess.check_output(cmd)

    return ani


def save_environment(filename, nb_points, nb_rbfs, sigma, centers, nb_samples,
                     w, costmap, starts, targets, paths):
    """ Save the environment with given demonstrations in a file """
    file = home + '/../data/environment/' + filename + '.npz'
    np.savez(file, nb_points=nb_points, nb_rbfs=nb_rbfs, sigma=sigma,
             centers=centers, nb_samples=nb_samples, w=w, costmap=costmap,
             starts=starts, targets=targets, paths=paths)
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
    centers = file['centers']
    return w, costmap, starts, targets, paths, centers


def load_environment_params(filename):
    """ Load the parametes of an environment from a file """
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

    np.savez(file, learning_rate=l._learning_rate,
             stepsize_scalar=l._stepsize_scalar,
             loss_scalar=l.instances[0]._loss_scalar,
             loss_stddev=l.instances[0]._loss_stddev,
             l2_regularizer=l.instances[0]._l2_regularizer,
             proximal_regularizer=l.instances[0]._proximal_regularizer,
             convergence=l.convergence)
    return file


def get_learch_params(directory):
    """ Return the hyperparameters of a LEARCH instance according to the values
        saved in the given file
    """
    file = np.load(directory)
    loss_scalar = file['loss_scalar']
    loss_stddev = file['loss_stddev']
    learing_rate = file['learning_rate']
    stepsize_scalar = file['stepsize_scalar']
    l2_regularizer = file['l2_regularizer']
    proximal_regularizer = file['proximal_regularizer']
    convergence = file['convergence']
    return learing_rate, stepsize_scalar, loss_scalar, loss_stddev, \
           l2_regularizer, proximal_regularizer, convergence


def set_learch_params(directory, l):
    """ Set the hyperparameters for a LEARCH instance according to the values
        saved in the given file
    """
    file = np.load(directory)
    l._learing_rate = file['learning_rate']
    l._stepsize_scalar = file['stepsize_scalar']
    l.convergence = file['convergence']
    for _, i in enumerate(l.instances):
        i._loss_scalar = file['loss_scalar']
        i._loss_stddev = file['loss_stddev']
        i._l2_regularizer = file['l2_regularizer']
        i._proximal_regularizer = file['proximal_regularizer']
    return l


def save_maxEnt_params(directory, m):
    """ Save the hyperparametes of a maxEnt instance in a file """
    file = directory + '.npz'
    np.savez(file, learning_rate=m._learning_rate, stepsize_scalar=
    m._stepsize_scalar, N=m.instances[0]._N, convergence=m.convergence)
    return file


def set_maxEnt_params(directory, m):
    """ Set the hyperparameters of a maxEnt instance according to the values
        saved in the given file
    """
    file = np.load(directory)
    m._learning_rate = file['learning_rate']
    m._stepsize_scalar = file['stepsize_scalar']
    m.convergence = file['convergence']
    for _, i in enumerate(m.instances):
        i._N = file['N']
    return m


def save_newAlg_params(directory, l):
    """ Save the hyperparametes of a LEARCH variant instance in a file """
    file = directory + '.npz'
    np.savez(file, learning_rate=l._learning_rate, stepsize_scalar=
    l._stepsize_scalar, N=l.instances[0]._N,
             loss_scalar=l.instances[0]._loss_scalar,
             loss_stddev=l.instances[0]._loss_stddev,
             l2_regularizer=l.instances[0]._l2_regularizer,
             proximal_regularizer=l.instances[0]._proximal_regularizer,
             convergence=l.convergence)
    return file


def set_newAlg_params(directory, l):
    """ Set the hyperparameters of a LEARCH variant instance according to
        the values saved in the given file
    """
    file = np.load(directory)
    l._learning_rate = file['learning_rate']
    l._stepsize_scalar = file['stepsize_scalar']
    l.convergence = file['convergence']
    for _, i in enumerate(l.instances):
        i._loss_scalar = file['loss_scalar']
        i._loss_stddev = file['loss_stddev']
        i._l2_regularizer = file['l2_regularizer']
        i._proximal_regularizer = file['proximal_regularizer']
        i._N = file['N']
    return l


def save_results(directory, maps, ex_paths, w_t, starts=None, targets=None,
                 paths=None):
    """ Save the results of one run """
    file = directory + '.npz'
    np.savez(file, maps=maps, optimal_paths=ex_paths, w=w_t, paths=paths,
             starts=starts, targets=targets)
    return file


def get_results(directory):
    """ Get the saved results of one run """
    file = np.load(directory, allow_pickle=True)
    maps = file['maps']
    ex_paths = file['optimal_paths']
    w_t = file['w']
    starts = file['starts']
    targets = file['targets']
    paths = file['paths']
    return maps, ex_paths, w_t, starts, targets, paths


FONTSIZE = 20
cmap = plt.get_cmap('viridis')
# plt.rcParams.update({'font.size': FONTSIZE})
#mlt.rcParams['font.size'] = FONTSIZE
