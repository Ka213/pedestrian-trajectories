import os
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

home = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, home)
sys.path.insert(0, home + os.sep + "../src/pyrieef")
sys.path.insert(0, home + os.sep + "..")
