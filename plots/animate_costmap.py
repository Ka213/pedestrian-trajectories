import common_import

from my_utils.output_costmap import *
from pyrieef.geometry.workspace import *

show_result = 'SAVE'

filename = 'learch_100samples'

workspace = Workspace()

w, costmap, starts, targets, paths, centers = load_environment("environment0")
maps, optimal_paths, _, starts, targets, paths = \
    get_results(home + '/../results/learning/' + filename + '.npz')

# Output animated 3D plot
animated_plot(maps, workspace, show_result,  # starts=starts, targets=targets,
              # paths=paths, #optimal_paths=optimal_paths,
              directory=home + '/../results/animations/' + filename)
