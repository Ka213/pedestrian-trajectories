from common_import import *

from my_utils.output_analysis import *

show_image = 'SHOW'
name = 'maxEnt_search_step size_5r_1l_15u'
name = 'results/learch_search_regularization_1r_0l_10u'

directory = home + '/../resutls/hyperparametersearch2D/' + name + '.npz'
directoryToSave = home + '/../results/hyperparametersearch2D/' + name + '.png'
plot_data(show_image, directory, directoryToSave)
