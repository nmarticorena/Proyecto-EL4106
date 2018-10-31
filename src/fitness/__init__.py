#!/usr/bin/env python

import ctypes
import os
import sys
import numpy as np

path = os.path.dirname(__file__)
distancedll = ctypes.CDLL(os.path.join(path, "distance.dll" if sys.platform.startswith("win") else "distance.so"))


_distance = distancedll.distance
_distance.restype = ctypes.c_double

def fitDistance(individual,objective):
    assert len(individual) == 4 and len(objective)== 3;
    if isinstance(individual, np.ndarray) and individual.dtype == np.float64 and \
            len(individual.shape)==1:
        # it's already a numpy array with the right features - go zero-copy
        indP = individual.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        # it's a list or something else - try to create a copy
        arr_t1 = ctypes.c_double * 4
        indP = arr_t1(*individual)
    if isinstance(objective, np.ndarray) and objective.dtype == np.float64 and \
            len(objective.shape)==1:
        # it's already a numpy array with the right features - go zero-copy
        objP = objective.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        # it's a list or something else - try to create a copy
        arr_t2 = ctypes.c_double * 3
        objP = arr_t2(*objective)
    return _distance(indP,objP)

