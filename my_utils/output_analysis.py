import common_import

from textwrap import wrap
import numpy as np
from matplotlib import cm
import matplotlib
import math
# print(matplotlib.get_backend())
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def plot_avg_over_runs(x, nb_runs, directory, loss=None, time=None,
                       nb_steps=None, prediction_time=None):
    """ Plot the given measurements over different number of samples,
        average over seeds
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if loss is not None:
        loss = np.average(loss, axis=0)
        if nb_runs > 1:
            y_stddev = np.std(loss, axis=0)
            ax.errorbar(x, loss, yerr=y_stddev, capsize=2, label='loss')
        else:
            ax.plot(x, loss, label='loss')
        plt.xlabel('loss')
        # ax.set_yscale('log')
        # ax.set_xticks(x[::2])

    if time is not None:
        time = np.average(time, axis=0)
        ax.plot(x, time, label='learning time in sec')
        plt.xlabel('learning time in sec')

    if prediction_time is not None:
        prediction_time = np.average(prediction_time, axis=0)
        ax.plot(x, prediction_time, label='prediction time in sec')
        plt.xlabel('prediction time in sec')

    if nb_steps is not None:
        steps = np.average(nb_steps, axis=0)
        ax.plot(x, steps, label='# of steps \nuntil convergence')
        plt.xlabel('# of steps \nuntil convergence')

    ax.legend(loc="upper right")
    # plt.legend()
    # plt.xticks(np.arange(x[0], x[-1] + (x[1] - x[0]),
    #                     math.ceil((len(x) / 10)) * (x[1] - x[0])))
    plt.xlabel('# number of demonstrations used')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # plt.title(
    #    '\n'.join(wrap('averaged over {} different environments'
    #                   .format(nb_runs), 60)))
    plt.savefig(directory)


def plot_data(show_image, directory, directoryToSave=None):
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
    zero = np.zeros(y.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    strings = directory.split('_')
    param = strings[2]
    learning = strings[0]
    if param == 'step size':
        ax.set_xlabel('step size scalar')
        ax.set_ylabel('learning rate')
    elif param == 'regularization':
        print("x ticks:", np.flipud(x1))
        print("y ticks:", np.flipud(x2))
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
                           facecolors=colors, shade=False, label='loss')
    surf.set_facecolor((0, 0, 0, 0))
    colors = cm.Blues(zero + 0.3)
    rcount, ccount, _ = colors.shape
    x_1, x_2 = np.meshgrid(x1, x2)
    z = ax.plot_surface(x_1, x_2, zero, rcount=rcount, ccount=ccount,
                        facecolors=colors, label='zero plane')
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


def add_subplot_plot(ax, x, y, name):
    for y_i, name_i in zip(y, name):
        ax.plot(x, y_i, label=name_i)


def compare_learning(directories, directory):
    """ plot results different predictions into the same graph """
    l_x = []
    l_nll = []
    l_test_loss = []
    l_training_loss = []
    l_test_edt = []
    l_training_edt = []
    l_costs = []
    l_nb_steps = []
    l_learning_time = []
    l_prediction_time = []
    l_name = []
    # Get each result
    for i, d in enumerate(directories):
        l = np.load(d)
        l_x.append(l['x'])
        l_nll.append(np.average(l['nll'], axis=0))
        l_test_loss.append(np.average(l['test_loss'], axis=0))
        l_training_loss.append(np.average(l['training_loss'], axis=0))
        l_test_edt.append(np.average(l['test_edt'], axis=0))
        l_training_edt.append(np.average(l['training_edt'], axis=0))
        l_costs.append(np.average(l['costs'], axis=0))
        l_nb_steps.append(np.average(l['nb_steps'], axis=0))
        l_learning_time.append(np.average(l['learning_time'], axis=0))
        l_prediction_time.append(np.average(l['prediction_time'], axis=0))
        l_name.append(d.split('/')[-2])
    # Plot each measurement in one graph
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(331)
    add_subplot_plot(ax, l_x[0], l_nll, l_name)
    ax.set_xlabel('# of demonstrations used')
    ax.set_ylabel('negative log likelihood')
    ax.legend(loc="upper right")
    ax.set_title("NLL")
    ax = fig.add_subplot(332)
    add_subplot_plot(ax, l_x[0], l_test_loss, l_name)
    ax.set_xlabel('# of demonstrations used')
    ax.set_ylabel('test loss')
    ax.legend(loc="upper right")
    ax.set_title("test loss")
    ax = fig.add_subplot(333)
    add_subplot_plot(ax, l_x[0], l_training_loss, l_name)
    ax.set_xlabel('# of demonstrations used')
    ax.set_ylabel('training loss')
    ax.legend(loc="upper right")
    ax.set_title("training loss")
    ax = fig.add_subplot(334)
    add_subplot_plot(ax, l_x[0], l_test_edt, l_name)
    ax.set_xlabel('# of demonstrations used')
    ax.set_ylabel('test euclidean distance transform')
    ax.legend(loc="upper right")
    ax.set_title("test euclidean distance transform")
    ax = fig.add_subplot(335)
    add_subplot_plot(ax, l_x[0], l_training_edt, l_name)
    ax.set_xlabel('# of demonstrations used')
    ax.set_ylabel('training euclidean distance transform')
    ax.legend(loc="upper right")
    ax.set_title("training euclidean distance transform")
    ax = fig.add_subplot(336)
    add_subplot_plot(ax, l_x[0], l_nb_steps, l_name)
    ax.set_xlabel('# of demonstrations used')
    ax.set_ylabel('iteration steps')
    ax.legend(loc="upper right")
    ax.set_title("iteration steps")
    ax = fig.add_subplot(337)
    add_subplot_plot(ax, l_x[0], l_learning_time, l_name)
    ax.set_xlabel('# of demonstrations used')
    ax.set_ylabel('learning time in sec')
    ax.legend(loc="upper right")
    ax.set_title("learning time")
    ax = fig.add_subplot(338)
    add_subplot_plot(ax, l_x[0], l_prediction_time, l_name)
    ax.set_xlabel('# of demonstrations used')
    ax.set_ylabel('inference time in sec')
    ax.legend(loc="upper right")
    ax.set_title("inference time")
    ax = fig.add_subplot(339)
    add_subplot_plot(ax, l_x[0], l_costs, l_name)
    ax.set_xlabel('# of demonstrations used')
    ax.set_ylabel('value difference of the costmaps')
    ax.legend(loc="upper right")
    ax.set_title("costmap difference")
    plt.tight_layout()
    plt.savefig(directory)


def plot_loss_fix_nbSamples(error, nb_samples, nb_runs, directory):
    """ Plot loss over different seeds for fixed number of samples """
    x = np.arange(nb_runs) + 1
    plt.figure()
    plt.plot(x, error)
    plt.ylabel('loss')
    plt.xlabel('run')
    plt.xticks(np.arange(1, nb_runs, math.ceil(len(nb_runs) / 10)))
    plt.title('loss over different seeds for {} samples'.format(nb_samples))
    plt.savefig(directory)


cmap = plt.get_cmap('viridis')
