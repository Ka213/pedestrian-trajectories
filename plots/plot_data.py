from common_import import *

from nn.dataset_sdf import *
from my_utils.output_costmap import *

show_result = 'SHOW'
nb_samples = 25
nb_env = 1000
learning = 'learch'
nn_steps = 100
# range of demonstrations
x = 0
y = 1
# workspace index
n = 1

name = learning + '_25samples_1000env_100'

dataset = 'sdf_data_' + str(nb_env) + '_40'  # + str(nb_samples)
foldername = '{}_{}samples_{}env_{}'.format(learning, nb_samples,
                                            nb_env, nn_steps)

workspaces = load_workspace_dataset(basename=dataset + '.hdf5')

box = EnvBox()
lims = workspaces[0].lims
box.dim[0] = lims[0][1] - lims[0][0]
box.dim[1] = lims[1][1] - lims[1][0]
box.origin[0] = box.dim[0] / 2.
box.origin[1] = box.dim[1] / 2.
workspace = Workspace(box)

starts = workspaces[n].starts
targets = workspaces[n].targets
demos = workspaces[n].demonstrations
costs = workspaces[n].costmap

paths = []
maps = []
for i in range(0, nb_samples - 1, 2):
    directory = home + '/../results/prediction_nn/' + name + '/maps' + \
                str(i + 1) + '.npz'
    data = np.load(directory, allow_pickle=True)

    learned_maps = data['maps']
    maps.append(learned_maps[-1][n])
    _, _, op = plan_paths(y - x, learned_maps[-1][n], workspace,
                          starts=starts[x:y], targets=targets[x:y])
    paths.append(op)

show_multiple(maps, [costs], Workspace(), show_result, starts=starts[x:y],
              targets=targets[x:y], paths=demos[x:y], ex_paths=paths,
              directory=home + '/../results/figures/nn_' + learning + '.pdf')
