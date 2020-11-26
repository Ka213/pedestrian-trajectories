from common_import import *

from my_utils.output_analysis import *

# directory
path = home + '/../results/prediction_rbfs_28/'

# list folders with results
directory_1 = path + "loss_aug_esf_1env_1-100samples_20predictions_average" \
              + "/results.npz"
directory_2 = path + "loss_aug_esf_5env_1-100samples_20predictions_[0 1 2 3 4]" \
              + "/results.npz"
directory_3 = path + "loss_aug_esf_10env_1-100samples_20predictions_[0 1 2 3 4 5 6 7 8 9]" \
              + "/results.npz"
directory_4 = path + "loss_aug_esf_20env_1-100samples_20predictions" \
              + "/results.npz"

directory_5 = path + "learch_20env_1-100samples_20predictions" \
              + "/results.npz"
directory_6 = path + "maxEnt_20env_1-100samples_20predictions" \
              + "/results.npz"
directory_7 = path + "loss_aug_esf_20env_1-100samples_20predictions" \
              + "/results.npz"
directory_8 = path + "oneVector_20env_1-100samples_20predictions_20_20" \
              + "/results.npz"
directory_9 = path + "random_20env_1-100samples_20predictions_20_20" \
              + "/results.npz"

directory_10 = path + "learch_25samples_250env_100" \
               + "/results.npz"
directory_11 = path + "loss_aug_esf_25samples_250env_100" \
               + "/results.npz"
directory_12 = path + "maxEnt_25samples_250env_100" \
               + "/results.npz"
directory_13 = path + "learch_25samples_500env_100" \
               + "/results.npz"
directory_14 = path + "loss_aug_esf_25samples_500env_100" \
               + "/results.npz"
directory_15 = path + "maxEnt_25samples_500env_100" \
               + "/results.npz"
directory_16 = path + "learch_25samples_1000env_100" \
               + "/results.npz"
directory_17 = path + "loss_aug_esf_25samples_1000env_100" \
               + "/results.npz"
directory_18 = path + "maxEnt_25samples_1000env_100" \
               + "/results.npz"

directory = path + "comparison_10env_2"

names_1 = ['1 environment', '5 environments', '10 environments',
           '20 environments']
names_2 = ['LEARCH', 'maximum entropy', 'LEARCH variation', 'one vector',
           'random']
names_nn = ['Deep-LEARCH 200 environments',
            'Deep-LEARCH variant 200 environments',
            'maximum entropy using CNNs 200 environments',
            'Deep-LEARCH 400 environments',
            'Deep-LEARCH variant 400 environments',
            'maximum entropy using CNNs 400 environments',
            'Deep-LEARCH 800 environments',
            'Deep-LEARCH variant 800 environments',
            'maximum entropy using CNNs 800 environments']
# output
compare_learning([directory_5, directory_6, directory_7,
                  directory_8, directory_9],
                 directory, names=names_2, title='-', single=False)
