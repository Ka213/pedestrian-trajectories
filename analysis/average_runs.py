from common_import import *

from my_utils.output_analysis import *
import datetime

# list files to average
file1 = 'maxEnt_1env_1-20samples_20predictions_[0]'
file2 = 'maxEnt_1env_1-20samples_20predictions_[1]'
file3 = 'maxEnt_1env_1-20samples_20predictions_[2]'
file4 = 'maxEnt_1env_1-20samples_20predictions_[3]'
file5 = 'maxEnt_1env_1-20samples_20predictions_[4]'
file6 = 'maxEnt_1env_1-20samples_20predictions_[5]'
file7 = 'maxEnt_1env_1-20samples_20predictions_[6]'
file8 = 'maxEnt_1env_1-20samples_20predictions_[7]'
file9 = 'maxEnt_1env_1-20samples_20predictions_[8]'
file10 = 'maxEnt_1env_1-20samples_20predictions_[9]'
directoryToSave = 'maxEnt_1env_1-20samples_20predictions_average'

files = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]

path = home + '/../results/prediction_rbfs_28/'

directory = path + files[0] + '/results.npz'
l = np.load(directory, allow_pickle=True)
x = len(l['x'])
nb_training = l['training_loss_l'].shape[1]
nb_test = l['test_loss_l'].shape[1]

l_x = []
l_test_nll = np.zeros((x, len(files), nb_test))
l_training_nll = np.zeros((x, len(files), nb_training))
l_test_loss_l = np.zeros((x, len(files), nb_test))
l_training_loss_l = np.zeros((x, len(files), nb_training))
l_test_loss_m = np.zeros((x, len(files), nb_test))
l_training_loss_m = np.zeros((x, len(files), nb_training))
l_test_edt = np.zeros((x, len(files), nb_test))
l_training_edt = np.zeros((x, len(files), nb_training))
l_test_costs = np.zeros((x, len(files), nb_test))
l_training_costs = np.zeros((x, len(files), nb_training))

l_nb_steps = []
l_learning_time = []
l_prediction_time = []
# Get each result
for i, d in enumerate(files):
    directory = path + d + '/results.npz'
    l = np.load(directory, allow_pickle=True)
    x = len(l['x'])
    l_x.append(l['x'])
    l_nb_steps.append(l['nb_steps'])
    l_learning_time.append(l['learning_time'])
    l_prediction_time.append(l['prediction_time'])
    l_test_loss_l[:, i] = l['test_loss_l']
    l_training_loss_l[:, i] = l['training_loss_l']
    l_test_loss_m[:, i] = l['test_loss_m']
    l_training_loss_m[:, i] = l['training_loss_m']
    l_test_nll[:, i] = l['test_nll']
    l_training_nll[:, i] = l['training_nll']
    l_test_edt[:, i] = l['test_edt']
    l_training_edt[:, i] = l['training_edt']
    l_training_costs[:, i] = l['training_costs']
    l_test_costs[:, i] = l['test_costs']

# define axis for correct standard deviation
if nb_training == 1:
    a = 2
else:
    a = 1
l_training_loss_l = np.average(l_training_loss_l, axis=a)
l_training_loss_m = np.average(l_training_loss_m, axis=a)
l_training_edt = np.average(l_training_edt, axis=a)
l_training_nll = np.average(l_training_nll, axis=a)
l_training_costs = np.average(l_training_costs, axis=a)

# define axis for correct standard deviation
if nb_test == 1:
    a = 2
else:
    a = 1
l_test_loss_l = np.average(l_test_loss_l, axis=a)
l_test_loss_m = np.average(l_test_loss_m, axis=a)
l_test_edt = np.average(l_test_edt, axis=a)
l_test_nll = np.average(l_test_nll, axis=a)
l_test_costs = np.average(l_test_costs, axis=a)

os.makedirs(path + directoryToSave)

file = open(path + directoryToSave + "/metadata.txt", "w")
file.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           + '\n')
file.close()
results = path + directoryToSave + '/results.npz'

np.savez(results, x=l_x[0], learning_time=np.average(l_learning_time, axis=0),
         prediction_time=np.average(l_prediction_time, axis=0),
         nb_steps=np.average(l_nb_steps, axis=0),
         training_loss_l=l_training_loss_l,
         training_loss_m=l_training_loss_m,
         training_edt=l_training_edt,
         training_costs=l_training_costs,
         training_nll=l_training_nll,
         test_loss_l=l_test_loss_l,
         test_loss_m=l_test_loss_m,
         test_nll=l_test_nll,
         test_edt=l_test_edt,
         test_costs=l_test_costs)

compare_learning([results], path + directoryToSave + '/output.pdf',
                 names=['average'])
