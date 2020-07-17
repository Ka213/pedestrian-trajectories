import common_import

from pyrieef.geometry.workspace import *
from costmap.costmap import *
from my_utils.my_utils import *
from my_utils.output import *


def test_policy_iteration():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    discount = 0.8
    show_result = 'SHOW'

    workspace = Workspace()
    np.random.seed(3)

    # Create costmap with rbfs
    w = np.random.random(nb_rbfs ** 2)
    centers = workspace.box.meshgrid_points(nb_rbfs)
    original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)

    P = get_transition_probabilities(original_costmap, nb_points)
    policy = policy_iteration(original_costmap, nb_points, discount, P)
    direction = np.argmax(policy, axis=1)
    print(direction)
    ancestor = np.zeros(nb_points ** 2)

    converter = CostmapToSparseGraph(original_costmap)
    converter.integral_cost = True
    graph = converter.convert()

    for i, d in enumerate(direction):
        s = converter.costmap_id(i)
        # up, down, right, up - right, down - right, left, up - left, down - left
        if d == 0:
            new_s = (s[0], s[1] - 1)
        elif d == 1:
            new_s = (s[0], s[1] + 1)
        elif d == 2:
            new_s = (s[0] + 1, s[1])
        elif d == 3:
            new_s = (s[0] + 1, s[1] - 1)
        elif d == 4:
            new_s = (s[0] + 1, s[1] + 1)
        elif d == 5:
            new_s = (s[0] - 1, s[1])
        elif d == 6:
            new_s = (s[0] - 1, s[1] - 1)
        else:
            new_s = (s[0] - 1, s[1] + 1)
        if converter.is_in_costmap(new_s[0], new_s[1]):
            s_new = converter.graph_id(new_s[0], new_s[1])
            ancestor[i] = s_new
    show(original_costmap, workspace, show_result, predecessors=ancestor)


def test_state_visitation_frequency_counts():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    N = 10
    show_result = 'SHOW'

    workspace = Workspace()
    np.random.seed(3)

    # Create costmap with rbfs
    w = np.random.random(nb_rbfs ** 2)
    centers = workspace.box.meshgrid_points(nb_rbfs)
    original_costmap = get_costmap(nb_points, centers, sigma, w, workspace)

    P = get_transition_probabilities(original_costmap, nb_points)
    frequency_counts = get_expected_edge_frequency(P, original_costmap, N,
                                                   nb_points)
    show(frequency_counts, workspace, show_result)


if __name__ == "__main__":
    show_result = 'SHOW'
    test_policy_iteration()
    test_state_visitation_frequency_counts()
