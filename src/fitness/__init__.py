#!/usr/bin/env python

import ctypes
import os
import sys
import numpy as np
import math
path = os.path.dirname(os.path.abspath(__file__))
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

#Test
if __name__ == '__main__':
    import math
    ind=np.array([1,0.5,1.1 ,-1],dtype='double')
    objective=[0.7,0.7,1]
    res=fitDistance(ind,objective)
    print(res)
    print(math.isclose(res,0.47195012030477))