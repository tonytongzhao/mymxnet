import numpy as np
import cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint

cdef inline int int_max(int a, int b) nogil: return a if a>=b else b

cdef inline int int_min(int a, int b) nogil: return b if a>=b else a

@cython.boundscheck(False)
@cython.wraparound(False)

def conv_bc01(np.ndarray[DTYPE_t, ndim=4] imgs, np.ndarray[DTYPE_t, ndim=4] filters, np.ndarray[DTYPE_t, ndim=4] convout):

