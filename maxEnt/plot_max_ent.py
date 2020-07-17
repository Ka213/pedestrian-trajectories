import common_import

from pyrieef.geometry.workspace import *
from pyrieef.geometry.interpolation import *
from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *
from pyrieef.learning.inverse_optimal_control import *
from maxEnt.max_ent import *
from my_utils.output import *
from costmap.costmap import *

show_result = 'SHOW'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 40
nb_steps = 4

# Path to current directory
home = os.path.abspath(os.path.dirname(__file__))

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)

# Create costmap with rbfs
w = np.random.random(nb_rbfs ** 2)
centers = workspace.box.meshgrid_points(nb_rbfs)
original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)

# Plan example trajectories
starts, targets, paths = plan_paths(nb_samples, original_costmap, workspace)

# Learn costmap
a = MaxEnt(nb_points, centers, sigma, paths, starts, targets, workspace)
for i in range(10):
    maps, w = a.one_step(i)
    print(w[-1])
    # Output learned costmaps
    show(maps[-1], workspace, show_result)
