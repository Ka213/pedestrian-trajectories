from common_import import *

from my_utils.output_analysis import *
import datetime

file1 = 'maxEnt_1env_5-100samples_100predictions_1_expGD'
file2 = 'maxEnt_1env_5-100samples_100predictions_2_expGD'
file3 = 'maxEnt_1env_5-100samples_100predictions_3_expGD'
file4 = 'maxEnt_1env_5-100samples_100predictions_4_expGD'
directoryToSave = 'maxEnt_4env_5-100samples_100predictions_expGD_average'

files = [file1, file2, file3, file4]

path = home + '/../results/prediction/'

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
# Get each result
for i, d in enumerate(files):
    directory = path + file1 + '/results.npz'
    l = np.load(directory, allow_pickle=True)
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

file = open(path + directoryToSave + "/metadata.txt", "w")
file.write("date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           + '\n')
file.close()
results = path + directoryToSave + '/results.npz'
np.savez(results, x=l_x[0], learning_time=np.average(l_learning_time, axis=0), prediction_time=
np.average(l_prediction_time, axis=0), nb_steps=np.average(l_nb_steps, axis=0),
         training_loss_l=np.average(l_training_loss_l, axis=0),
         training_loss_m=np.average(l_training_loss_m, axis=0), training_edt=np.average(l_training_edt, axis=0),
         training_costs=np.average(l_training_costs, axis=0), training_nll=np.average(l_training_nll, axis=0),
         test_loss_l=np.average(l_test_loss_l, axis=0), test_loss_m=np.average(l_test_loss_m, axis=0), test_nll=
         np.average(l_test_nll, axis=0), test_edt=np.average(l_test_edt, axis=0),
         test_costs=np.average(l_test_costs, axis=0))
