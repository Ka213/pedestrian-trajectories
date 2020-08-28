from common_import import *

from my_utils.output_analysis import *

path = home + '/../results/prediction/'

directory_1 = path + "/learch_1runs_1-100samples_100predictions" \
              + "/results.npz"
directory_2 = path + "/learch_TrainOn2CostmapsTestOnOne" \
              + "/results.npz"
directory_3 = path + "/learch_TrainOnOneCostmapTestOnOtherCostmap" \
              + "/results.npz"
directory = path + "/comparison_learch.png"

compare_learning([directory_1, directory_2, directory_3], directory)
