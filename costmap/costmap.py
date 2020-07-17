import common_import

import numpy as np
from scipy.interpolate import Rbf


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


def get_costmap(nb_points, centers, sigma, w, workspace):
    """ Returns the costmap of RBFs"""
    costmap = np.tensordot(w,
                           get_phi(nb_points, centers, sigma, workspace),
                           axes=1)
    return costmap
