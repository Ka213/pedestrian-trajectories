import common_import

from my_utils.output_costmap import *
from pyrieef.geometry.workspace import *

show_result = 'SHOW'

filename = 'maxEnt'
directory = home + '/../results/learning/'
workspace = Workspace()

maps, ex_paths, w_t, starts, targets, paths = get_results(directory +
                                                          filename +
                                                          '.npz')

# Output animated 3D plot
animated_plot(maps, workspace, show_result, starts=starts, targets=targets,
              ex_paths=ex_paths, directory=home + '/../results/animations/' +
                                           filename)
