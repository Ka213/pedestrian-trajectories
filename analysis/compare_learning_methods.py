from common_import import *

from my_utils.output_analysis import *

path = home + '/../results/prediction/'

directory_learch = path + "/learch_1runs_1-100samples_100predictions" \
                   + "/results.npz"
directory_maxEnt = path + "/maxEnt_1runs_1-100samples_100predictions" \
                   + "/results.npz"
directory = path + "/comparison_100_samples_100predictions.png"

compare_learning_methods(directory_learch, directory_maxEnt, directory)
