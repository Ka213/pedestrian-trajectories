from common_import import *

from my_utils.output_analysis import *

path = home + '/../results/prediction_nn/'

directory_1 = path + "learch_5env_1-20samples_20predictions" \
              + "/results.npz"
directory_2 = path + "maxEnt_5env_1-20samples_20predictions" \
              + "/results.npz"
directory_3 = path + "loss_aug_esf_5env_1-20samples_20predictions_" \
              + "/results.npz"
directory_4 = path + "oneVector_5env_1-20samples_20predictions" \
              + "/results.npz"
directory_5 = path + "random_5env_1-20samples_20predictions" \
              + "/results.npz"
directory_6 = path + "maxEnt_5env_1-100samples_20predictions" \
              + "/results.npz"
directory_7 = path + "onlyPushDown_5env_5-100samples_50predictions" \
              + "/results.npz"
directory_8 = path + "learch_20samples_5steps" \
              + "/results.npz"
directory = path + "deep-learch"
names = ['deep learch', 'maximum entropy', 'learch variation', 'one Vector', 'random',
         'only push down 5 env', 'only push down 5 env']
compare_learning([directory_8], directory, names=names, title='Comparison',
                 single=False)
