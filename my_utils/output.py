import common_import

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
    pixel_map = workspace.pixel_map(costmaps[0].shape[0])
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


def plot_error_avg(error, nb_steps, x, nb_runs, directory):
    """ Plot error over different number of samples, average over seeds """
    y = np.average(error, axis=0)
    s = np.average(nb_steps, axis=0)
    zero = np.zeros((len(x)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if nb_runs > 1:
        y_stddev = np.std(error, axis=0)
        ax.errorbar(x, y, yerr=y_stddev, capsize=2, label='loss')
    else:
        ax.plot(x, y, label='error')
    ax.plot(x, s, label='number of steps \nuntil convergence')
    ax.plot(x, zero, '--')
    ax.legend(loc="upper right")
    plt.legend()
    plt.xticks(np.arange(x[0], x[-1] + (x[1] - x[0]),
                         math.ceil((len(x) / 10)) * (x[1] - x[0])))
    # ax.set_xscale('log')
    # ax.set_xticks(x[::2])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title(
        '\n'.join(wrap('loss averaged '
                       'over {} different environments'.format(nb_runs), 60)))
    plt.savefig(directory)


def plot_from_file(show_image, directory, directoryToSave):
    """ Read data from file and plot it in 3D graph """
    matplotlib.use('Qt5Agg')
    print(matplotlib.get_backend())
    file = np.load(directory)
    error = file['loss']
    y = np.average(error, axis=0)
    nb_steps = file['nb_steps']
    s = np.average(nb_steps, axis=0)
    x1 = file['x1']
    x2 = file['x2']
    zero = np.zeros((len(x1), len(x2)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    strings = directory.split('_')
    param = strings[2]
    learning = strings[0]
    if param == 'step size':
        ax.set_xlabel('step size scalar')
        ax.set_ylabel('learning rate')
    elif param == 'regularization':
        print(x1)
        print(x2)
        x1 = np.arange(len(x1))
        x2 = np.arange(len(x2))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('proximal regularization')
        ax.set_ylabel('l2 regularization')
    elif param == 'loss':
        ax.set_xlabel('loss scalar')
        ax.set_ylabel('loss stddev')
    x_1, x_2 = np.meshgrid(x1, x2)
    norm = plt.Normalize(y.min(), y.max())
    colors = cm.viridis(norm(y))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(x_1, x_2, y, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False, label='error')
    surf.set_facecolor((0, 0, 0, 0))
    nb_runs = file['nb_runs']
    norm = plt.Normalize(s.min(), s.max())
    colors = cm.Reds(norm(s))
    rcount, ccount, _ = colors.shape
    # numbers = ax.plot_surface(x_1, x_2, s, rcount=rcount, ccount=ccount,
    #                          facecolors=colors, shade=False,
    #                          label='number of steps \nuntil convergence')
    # numbers.set_facecolor((0, 0, 0, 0))
    # ax.plot_surface(x_1, x_2, zero)
    ax.set_zlabel('loss')
    plt.title(
        '\n'.join(wrap('loss averaged '
                       'over {} different environments'.format(nb_runs), 60)))
    if show_image == 'SHOW':
        plt.show()
    elif show_image == 'SAVE':
        plt.savefig(directoryToSave)


def plot_error_fix_nbsamples(error, nb_samples, nb_runs, directory):
    """ Plot error over different seeds for fixed number of samples """
    x = np.arange(nb_runs) + 1
    plt.figure()
    plt.plot(x, error)
    plt.ylabel('loss')
    plt.xlabel('runs')
    plt.xticks(np.arange(1, nb_runs, math.ceil(len(nb_runs) / 10)))
    plt.title('loss over different seeds for {} samples'.format(nb_samples))
    plt.savefig(directory)


cmap = plt.get_cmap('viridis')
