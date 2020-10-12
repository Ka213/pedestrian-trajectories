import common_import
from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *
from my_utils.environment import *
from my_utils.my_utils import *
from my_utils.output_costmap import *


def test_policy_iteration():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    discount = 0.8
    show_result = 'SHOW'

    workspace = Workspace()
    np.random.seed(1)
    # Create random costmap
    w, original_costmap, starts, targets, paths, centers = \
        create_rand_env(nb_points, nb_rbfs, sigma, 0, workspace)

    P = get_transition_probabilities(original_costmap)
    policy = policy_iteration(original_costmap, nb_points, discount, P)
    direction = np.argmax(policy, axis=1)
    print(direction)
    ancestor = np.zeros(nb_points ** 2)

    converter = CostmapToSparseGraph(original_costmap)
    converter.integral_cost = True
    graph = converter.convert()

    for i, d in enumerate(direction):
        s = converter.costmap_id(i)
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


def test_expected_edge_frequency():
    show_result = 'SHOW'
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 300
    N = 5

    workspace = Workspace()
    np.random.seed(1)
    # Create random costmap
    w, original_costmap, starts, targets, paths, centers = \
        create_rand_env(nb_points, nb_rbfs, sigma, nb_samples,
                        workspace)
    Phi = get_phi(nb_points, centers, sigma, workspace)

    P = get_transition_probabilities(original_costmap)
    D = get_expected_edge_frequency(P, original_costmap, N, nb_points, targets,
                                    paths, workspace)
    D = - D - np.min(-D)
    f = np.tensordot(Phi, D)

    print("expected: ", (np.absolute(f - w)).sum())
    assert (np.absolute(f - w)).sum() < len(w) * 0.5

    phi = get_phi(nb_points, centers, sigma, workspace)
    map = get_costmap(phi, f)
    show_multiple([map], [original_costmap], workspace, show_result,
                  # starts=starts, targets=targets, paths=paths,
                  title="expected state visitation frequency")


def test_get_empirical_feature_count():
    nb_points = 40
    nb_rbfs = 5
    sigma = 0.1
    nb_samples = 200

    workspace = Workspace()
    np.random.seed(1)
    # Create random costmap
    w, original_costmap, starts, targets, paths, centers = \
        create_rand_env(nb_points, nb_rbfs, sigma, nb_samples,
                        workspace)
    Phi = get_phi(nb_points, centers, sigma, workspace)

    f = get_empirical_feature_count(paths, Phi)
    f = - f - np.min(- f)
    print("empirical features:", (np.absolute(f - w)).sum())
    assert (np.absolute(f - w)).sum() < len(w) * 0.5

    phi = get_phi(nb_points, centers, sigma, workspace)
    map = get_costmap(phi, f)
    show_multiple([map], [original_costmap], workspace, show_result,
                  # starts=starts, targets=targets, paths=paths,
                  title="empirical feature counts")

if __name__ == "__main__":
    show_result = 'SHOW'
    test_expected_edge_frequency()
    test_policy_iteration()
    test_get_empirical_feature_count()
