from common_import import *

from textwrap import wrap
import numpy as np
from matplotlib import cm, gridspec
import matplotlib
from matplotlib.ticker import MaxNLocator
import math
# print(matplotlib.get_backend())
import matplotlib.pyplot as plt


def plot_avg_over_runs(x, nb_runs, directory, loss=None, time=None,
                       nb_steps=None, prediction_time=None):
    """ Plot the given measurements over different number of samples,
        average over seeds
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if loss is not None:
        if nb_runs > 1:
            y_stddev = np.std(loss, axis=1)
            y = np.average(loss, axis=1)
            ax.errorbar(x, y, yerr=y_stddev, capsize=2, label='loss')
        else:
            ax.plot(x, loss, label='loss')
        plt.xlabel('loss')
        # ax.set_yscale('log')
        # ax.set_xticks(x[::2])

    if time is not None:
        ax.plot(x, time, label='learning time in sec')
        plt.xlabel('learning time in sec')

    if prediction_time is not None:
        ax.plot(x, prediction_time, label='prediction time in sec')
        plt.xlabel('prediction time in sec')

    if nb_steps is not None:
        ax.plot(x, nb_steps, label='# of steps \nuntil convergence')
        plt.xlabel('# of steps \nuntil convergence')

    ax.legend(loc="upper right")
    # plt.legend()
    # plt.xticks(np.arange(x[0], x[-1] + (x[1] - x[0]),
    #                     math.ceil((len(x) / 10)) * (x[1] - x[0])))
    plt.xlabel('# number of demonstrations per environment')
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
    # ax.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y'])))
    for x_i, y_i, name_i in zip(x, y, name):
        y_i = np.vstack(y_i)
        y_stddev = np.std(y_i, axis=1)
        y_i = np.average(y_i, axis=1)
        ax.fill_between(x_i, y_i + y_stddev, y_i - y_stddev, alpha=0.5)
        ax.plot(x_i, y_i, label=name_i)
        ax.xaxis.get_major_locator().set_params(integer=True)


def compare_learning(directories, directory, names=None, title=None,
                     single=False):
    """ plot results different predictions into the same graph """
    l_x = []
    l_test_nll = []
    l_training_nll = []
    l_test_loss_l = []
    l_training_loss_l = []
    l_test_loss_m = []
    l_training_loss_m = []
    l_test_edt = []
    l_training_edt = []
    l_training_costs = []
    l_test_costs = []
    l_nb_steps = []
    l_learning_time = []
    l_prediction_time = []
    l_name = names
    # Get each result
    for i, d in enumerate(directories):
        l = np.load(d, allow_pickle=True)
        l_x.append(l['x'])
        l_nb_steps.append(l['nb_steps'])
        l_learning_time.append(l['learning_time'])
        l_prediction_time.append(l['prediction_time'])
        l_test_loss_l.append(l['test_loss_l'])
        l_training_loss_l.append(l['training_loss_l'])
        l_test_loss_m.append(l['test_loss_m'])
        l_training_loss_m.append(l['training_loss_m'])
        l_test_nll.append(l['test_nll'])
        l_training_nll.append(l['training_nll'])
        l_test_edt.append(l['test_edt'])
        l_training_edt.append(l['training_edt'])
        l_training_costs.append(l['training_costs'])
        l_test_costs.append(l['test_costs'])

        if names is None:
            l_name.append(d.split('/')[-2])
    # Plot each measurement in one graph
    rows = 7
    cols = 2
    axes = []
    figures = []
    if single:
        for i in range(13):
            fig = plt.figure(figsize=(7, 7))
            figures.append(fig)
            ax = fig.add_subplot(111)
            axes.append(ax)
    else:
        fig = plt.figure(figsize=(cols * 7, rows * 7))
        spec2 = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)
        figures.append(fig)
        for i in range(rows):
            for j in range(cols):
                if i * j == 6:
                    break
                axes.append(fig.add_subplot(spec2[i, j]))

    if title is not None:
        plt.suptitle(title, fontsize=20, y=1)

    ax = axes[0]
    add_subplot_plot(ax, l_x, l_training_loss_l, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('Learch training loss')
    ax.legend(loc="lower right")
    ax.set_title("Learch training loss")
    ax.set_ylim([-1, 0.1])

    ax = axes[1]
    add_subplot_plot(ax, l_x, l_test_loss_l, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('Learch test loss')
    ax.legend(loc="lower right")
    ax.set_title("Learch test loss")
    ax.set_ylim([-0.3, 0.1])

    ax = axes[2]
    add_subplot_plot(ax, l_x, l_training_loss_m, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('maximum entropy training loss')
    ax.legend(loc="upper right")
    ax.set_title("maximum entropy training loss")
    ax.set_ylim([0, 10])

    ax = axes[3]
    add_subplot_plot(ax, l_x, l_test_loss_m, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('maximum entropy test loss')
    ax.legend(loc="upper right")
    ax.set_title("maximum entropy test loss")
    ax.set_ylim([0, 10])

    ax = axes[4]
    add_subplot_plot(ax, l_x, l_training_edt, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('training euclidean distance transform')
    ax.legend(loc="upper right")
    ax.set_title("training euclidean distance transform")
    ax.set_ylim([0, 2])

    ax = axes[5]
    add_subplot_plot(ax, l_x, l_test_edt, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('test euclidean distance transform')
    ax.legend(loc="upper right")
    ax.set_title("test euclidean distance transform")
    ax.set_ylim([0, 2])

    ax = axes[6]
    add_subplot_plot(ax, l_x, l_training_nll, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('training negative log likelihood')
    ax.legend(loc="upper right")
    ax.set_title("training NLL")
    ax.set_ylim([0, 2])

    ax = axes[7]
    add_subplot_plot(ax, l_x, l_test_nll, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('test negative log likelihood')
    ax.legend(loc="upper right")
    ax.set_title("test NLL")
    ax.set_ylim([0, 2])

    ax = axes[8]
    add_subplot_plot(ax, l_x, l_training_costs, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('value difference of the costmaps')
    ax.legend(loc="upper right")
    ax.set_title("training costmap difference")
    ax.set_ylim([0, 0.5])

    ax = axes[9]
    add_subplot_plot(ax, l_x, l_test_costs, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('value difference of the costmaps')
    ax.legend(loc="upper right")
    ax.set_title("test costmap difference")
    ax.set_ylim([0, 0.5])

    ax = axes[10]
    add_subplot_plot(ax, l_x, l_learning_time, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('learning time in sec')
    ax.legend(loc="upper right")
    ax.set_title("learning time")
    # ax.set_ylim([0, 1000])

    ax = axes[11]
    add_subplot_plot(ax, l_x, l_prediction_time, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('inference time in sec')
    ax.legend(loc="upper right")
    ax.set_title("inference time")
    # ax.set_ylim([0, 200])

    ax = axes[12]
    add_subplot_plot(ax, l_x, l_nb_steps, l_name)
    ax.set_xlabel('# of demonstrations used per environment')
    ax.set_ylabel('iteration steps')
    ax.legend(loc="upper right")
    ax.set_title("iteration steps")
    # ax.set_ylim([0, 60])
    if single:
        if not os.path.exists(directory):
            os.makedirs(directory)
        names = ['learch_train_loss', 'learch_test_loss', 'maxEnt_train_loss',
                 'maxEnt_test_loss', 'edt_train_loss', 'edt_test_loss',
                 'nll_train_loss', 'nll_test_loss', 'costs_train_loss',
                 'costs_test_loss', 'learning_time', 'inference time', 'steps']
        for f, n in zip(figures, names):
            f.savefig(directory + '/' + n + '.pdf', bbox_inches='tight')
    else:
        fig.tight_layout()
        plt.savefig(directory + '.pdf')


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
