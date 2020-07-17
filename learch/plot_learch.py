import common_import

from pyrieef.geometry.interpolation import *
from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *
from pyrieef.learning.inverse_optimal_control import *
from costmap.costmap import *
from my_utils.my_utils import *
from my_utils.output import *
from learch import *

show_result = 'SHOW'
nb_points = 40
nb_rbfs = 5
sigma = 0.1
nb_samples = 6
nb_steps = 5

# Path to current directory
home = os.path.abspath(os.path.dirname(__file__))

workspace = Workspace()
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(3)

# Create costmap with rbfs
w = np.random.random(nb_rbfs ** 2)
centers = workspace.box.meshgrid_points(nb_rbfs)
original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)

# Plan example trajectories
starts, targets, paths = plan_paths(nb_samples, original_costmap, workspace)

# Learn costmap
l = Learch2D(nb_points, centers, sigma, paths, starts, targets, workspace)

maps, optimal_paths, w = l.solve()  # n_step(nb_steps)

# show(maps[-1], workspace, show_result)
show_multiple([maps[-1]], original_costmap, workspace, show_result)
show_multiple(maps[::10], original_costmap, workspace, show_result,
              directory=home + "test1.pdf")
# show_iteration(maps, original_costmap, workspace, show_result, starts=starts,
#               targets=targets, paths=paths, optimal_paths=optimal_paths)
