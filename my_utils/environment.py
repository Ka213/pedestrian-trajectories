import common_import

import time
from scipy.interpolate import Rbf

from pyrieef.geometry.workspace import *
from pyrieef.graph.shortest_path import *

# For environment created from radial basis functions
def get_rbf(nb_points, center, sigma, workspace):
    """ Returns a radial basis function phi_i as map
        phi_i = exp(-(x-center/sigma)**2)
    """
    X, Y = workspace.box.meshgrid(nb_points)
    rbf = Rbf(center[0], center[1], 1, function='gaussian', epsilon=sigma)
    map = rbf(X, Y)

    return map


def get_phi(nb_points, centers, sigma, workspace):
    """ Returns the radial basis functions as vector """
    rbfs = []
    for i, center in enumerate(centers):
        rbfs.append(get_rbf(nb_points, center, sigma, workspace))
    phi = np.stack(rbfs)

    return phi


def get_costmap(phi, w):
    """ Returns the costmap of RBFs"""
    costmap = np.tensordot(w, phi, axes=1)
    return costmap


def create_rand_env(nb_points, nb_rbfs, sigma, nb_samples, workspace):
    """ Returns a random environment """
    # Create costmap with rbfs
    w = np.random.random(nb_rbfs ** 2)
    centers = workspace.box.meshgrid_points(nb_rbfs)
    phi = get_phi(nb_points, centers, sigma, workspace)
    costmap = get_costmap(phi, w)

    # Plan example trajectories
    starts, targets, paths = plan_paths(nb_samples, costmap, workspace)

    return w, costmap, starts, targets, paths, centers


def create_env_rand_centers(nb_points, nb_rbfs, sigma, nb_samples, workspace):
    """ Returns a random environment """
    # Create costmap with rbfs
    # w = [0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548, 0.64589411,
    #     0.43758721, 0.891773, 0.96366276, 0.38344152, 0.79172504, 0.52889492,
    #     0.56804456, 0.92559664, 0.07103606, 0.0871293, 0.0202184, 0.83261985,
    #     0.77815675, 0.87001215, 0.97861834, 0.79915856, 0.46147936, 0.78052918,
    #    0.11827443]
    w = np.random.random(nb_rbfs ** 2)
    centers = []
    for i in range(nb_rbfs ** 2):
        centers.append(sample_collision_free(workspace))
    phi = get_phi(nb_points, centers, sigma, workspace)
    costmap = get_costmap(phi, w)

    # Plan example trajectories
    starts, targets, paths = plan_paths(nb_samples, costmap, workspace)

    return w, costmap, starts, targets, paths, centers


def plan_paths(nb_samples, costmap, workspace, starts=None, targets=None,
               average_cost=False):
    """ Plan example trajectories
        either with random or fixed start and target state
    """
    #costmap += 1 # if the costmap only has zeros
    converter = CostmapToSparseGraph(costmap, average_cost)
    converter.integral_cost = True
    graph = converter.convert()
    pixel_map = workspace.pixel_map(costmap.shape[0])

    paths = []
    # Choose starts of the trajectory randomly
    if starts is None:
        starts = []
        for i in range(nb_samples):
            s_w = sample_collision_free(workspace)
            starts.append(s_w)
    # Choose targets of the trajectory randomly
    if targets is None:
        targets = []
        for i in range(nb_samples):
            t_w = sample_collision_free(workspace)
            targets.append(t_w)
    # Plan path
    for s_w, t_w in zip(starts, targets):
        s = pixel_map.world_to_grid(s_w)
        t = pixel_map.world_to_grid(t_w)
        try:
            # print("planning...")
            time_0 = time.time()
            # Compute the shortest path between the start and the target
            path = converter.dijkstra_on_map(costmap, s[0], s[1], t[0], t[1])
            paths.append(path)
        except Exception as e:
            print("Exception while planning a path")
            print(e)
            paths.append([(s[0], s[1])])
            continue
        # print("took t : {} sec.".format(time.time() - time_0))

    return starts, targets, paths


'''# For environment created by pixel permutation
def get_phi(nb_points, perm, sigma, workspace):
    """ Returns the permutation matrix """
    phi = perm.reshape((len(perm), nb_points, nb_points))
    return phi


def get_costmap(p, w):
    """ Returns the costmap """
    costmap = np.tensordot(w, p, axes=1)
    return costmap


def create_rand_pixel_env(nb_points, nb_rbfs, sigma, nb_samples, workspace):
    """ Returns a random environment """
    w = [0.44095047, 0.86059755, 0.16534147, 0.50704906, 0.79370644,
       0.67861966, 0.36662587, 0.57924415, 0.7362189 , 0.53337372,
       0.25348289, 0.44730234, 0.64484374, 0.93707173, 0.27360015,
       0.26348166, 0.01103235, 0.28785474, 0.43895929, 0.66586206,
       0.70470118, 0.92991973, 0.98524787, 0.43928033, 0.25491865,
       0.95270668, 0.02281501, 0.85025526, 0.46422741, 0.71163176,
       0.32267371, 0.70173982, 0.62725134, 0.4862258 , 0.50779675,
       0.36401991, 0.5597422 , 0.52583863, 0.91056596, 0.69923484,
       0.77441562, 0.72262876, 0.29261632, 0.38714312, 0.02306075,
       0.62048211, 0.22509244, 0.56941218, 0.80689298, 0.4835981 ,
       0.21504135, 0.74225926, 0.58224095, 0.8508989 , 0.19208861,
       0.21720129, 0.59716553, 0.41472829, 0.68767836, 0.85153372,
       0.4602322 , 0.85931127, 0.07562836, 0.8121574 , 0.43756112,
       0.58398506, 0.55925065, 0.14200852, 0.32167184, 0.82391926,
       0.05880268, 0.49804503, 0.13140375, 0.39789447, 0.45184212,
       0.65634953, 0.34366859, 0.77631678, 0.35738354, 0.69862332,
       0.1870895 , 0.88559032, 0.71292397, 0.91577944, 0.90021697,
       0.58012855, 0.26004607, 0.33096674, 0.5316096 , 0.09841921,
       0.37727633, 0.44160368, 0.45405491, 0.44725748, 0.4993083 ,
       0.0721103 , 0.08910385, 0.9709088 , 0.42513737, 0.96776692,
       0.76996142, 0.76815777, 0.29516144, 0.52877612, 0.34998186,
       0.44148932, 0.81804832, 0.82500826, 0.50217476, 0.06437734,
       0.82809988, 0.56841577, 0.15228379, 0.37915813, 0.8415831 ,
       0.17170326, 0.23715198, 0.85240884, 0.1268192 , 0.0799646 ,
       0.00229975, 0.18792687, 0.66071067, 0.29968979, 0.6226879 ,
       0.43374839, 0.47024393, 0.72391398, 0.22764012, 0.72916833,
       0.89415917, 0.62142609, 0.16476269, 0.86642221, 0.46767014,
       0.35756302, 0.45725133, 0.23670058, 0.99611585, 0.82136948,
       0.71388332, 0.28448847, 0.29335017, 0.29648344, 0.24674801,
       0.2538903 , 0.93445001, 0.29781556, 0.03327016, 0.93251122,
       0.63285844, 0.11751533, 0.05800966, 0.22514263, 0.77556974,
       0.54548414, 0.02156026, 0.79950674, 0.83464665, 0.33047608,
       0.69759831, 0.62435301, 0.64797299, 0.38382126, 0.80526045,
       0.73138433, 0.96348185, 0.25244928, 0.36194918, 0.90974863,
       0.70996787, 0.15591582, 0.55923212, 0.21860456, 0.20704878,
       0.65299102, 0.62529427, 0.50119088, 0.325919  , 0.50984611,
       0.9190863 , 0.99535651, 0.54464268, 0.79401415, 0.73741127,
       0.21446048, 0.15981527, 0.26849023, 0.49818776, 0.88429422,
       0.39668092, 0.33050993, 0.47255186, 0.94059061, 0.99501128,
       0.4312417 , 0.83929302, 0.31343469, 0.66070258, 0.3104778 ,
       0.28089536, 0.09297513, 0.51854191, 0.8991415 , 0.59820051,
       0.32923849, 0.75622499, 0.2484958 , 0.29318761, 0.1256868 ,
       0.89593375, 0.69390774, 0.2463027 , 0.25254777, 0.75571993,
       0.50351133, 0.22508592, 0.86879476, 0.28164249, 0.33041176,
       0.41978424, 0.02535909, 0.91884082, 0.07254735, 0.41492098,
       0.06840496, 0.63664411, 0.86692237, 0.98540807, 0.38267087,
       0.68788971, 0.65322106, 0.03548336, 0.1315558 , 0.00356181,
       0.15633331, 0.30827069, 0.1555597 , 0.40071964, 0.99363705,
       0.34681629, 0.32389474, 0.67713141, 0.68894372, 0.87296143,
       0.59220893, 0.13751315, 0.56535997, 0.80751433, 0.47275796,
       0.06859122, 0.80016749, 0.06741842, 0.09625369, 0.14373467,
       0.24017337, 0.95470299, 0.1304836 , 0.5482274 , 0.93360795,
       0.15023781, 0.5643359 , 0.1501826 , 0.13396422, 0.54027047,
       0.6842644 , 0.91122532, 0.82679441, 0.45630147, 0.01194064,
       0.51125308, 0.94625501, 0.39213768, 0.62978349, 0.82780464,
       0.9496214 , 0.61693523, 0.78057371, 0.58101492, 0.84802396,
       0.37072984, 0.60819638, 0.6809178 , 0.24639638, 0.79817531,
       0.08369935, 0.31135992, 0.10986938, 0.03527231, 0.89979013,
       0.37500441, 0.76143064, 0.06999081, 0.51605644, 0.92483472,
       0.60382636, 0.86640616, 0.7066625 , 0.46820365, 0.07471079,
       0.79013556, 0.37727939, 0.74588798, 0.86303102, 0.58299077,
       0.67540536, 0.98355616, 0.64760917, 0.21010636, 0.13212563,
       0.40352625, 0.11916959, 0.32190978, 0.81902719, 0.12713056,
       0.57903271, 0.71869516, 0.79170758, 0.36201615, 0.10693385,
       0.8941601 , 0.58492003, 0.0718035 , 0.62030537, 0.65710941,
       0.28745587, 0.24112625, 0.44078609, 0.94230617, 0.77590523,
       0.0237812 , 0.44886412, 0.12173943, 0.7342268 , 0.12939992,
       0.99935958, 0.93196484, 0.00579961, 0.81972197, 0.05446951,
       0.33821922, 0.13982936, 0.51919739, 0.23655371, 0.8579329 ,
       0.91111425, 0.08675984, 0.07742012, 0.97624753, 0.19022383,
       0.46111194, 0.63729182, 0.03283121, 0.20777185, 0.6895049 ,
       0.54024066, 0.48456989, 0.69676027, 0.73567544, 0.52340587,
       0.26462577, 0.03992262, 0.88691633, 0.76190064, 0.23681409,
       0.84447875, 0.58775249, 0.78812117, 0.24128469, 0.92703492,
       0.07796555, 0.81386387, 0.11077775, 0.76736341, 0.071415  ,
       0.63339138, 0.21805433, 0.74186803, 0.59832692, 0.77584   ,
       0.42180708, 0.79654561, 0.38392836, 0.69787456, 0.52451622,
       0.38158176, 0.05191342, 0.27251885, 0.05864333, 0.0088845 ,
       0.51257505, 0.80231263, 0.0395257 , 0.19706027, 0.86247297,
       0.72204997, 0.06876607, 0.33042619, 0.18659513, 0.49725427]
    #w = np.random.random(nb_points ** 2)
    map = np.random.permutation(nb_rbfs ** 2)
    p = np.zeros((nb_points ** 2, nb_points ** 2))
    for i, j in enumerate(map):
        p[i, j] = 1
    phi = get_phi(nb_points, p, sigma, workspace)
    costmap = get_costmap(phi, w)

    # Plan example trajectories
    starts, targets, paths = plan_paths(nb_samples, costmap, workspace)

    return w, costmap, starts, targets, paths, p
'''
