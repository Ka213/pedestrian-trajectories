from common_import import *

from my_utils.output import *

show_image = 'SHOW'
# name = 'maxEnt_search_step size_1r_0l_20u'
name = 'learch_search_loss_1r_0l_20u'

directory = home + '/../data/' + name + '.npz'
directoryToSave = home + '/../figures/' + name + '.png'
plot_from_file(show_image, directory, directoryToSave)
