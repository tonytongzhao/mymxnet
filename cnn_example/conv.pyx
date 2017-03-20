import numpy as np
import cython

cimport numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint

cdef inline int int_max(int a, int b) nogil: return a if a>=b else b

cdef inline int int_min(int a, int b) nogil: return b if a>=b else a

@cython.boundscheck(False)
@cython.wraparound(False)

def conv_bc01(np.ndarray[DTYPE_t, ndim=4] imgs, np.ndarray[DTYPE_t, ndim=4] filters, np.ndarray[DTYPE_t, ndim=4] convout):
    #imgs have shape, (n_imgs, n_channels_in, img_h, img_w)
    #filters have shape, (n_channels_in, n_channels_out, img_h, img_w)
    #convout have shape, (n_imgs, n_channels_out, img_h, img_w)
    cdef uint n_imgs=imgs.shape[0]
    cdef uint img_h=imgs.shape[2]
    cdef uint img_w=imgs.shape[3]

    cdef uint n_channels_in= filters.shape[0]
    cdef uint n_channels_out= filters.shape[1]

    cdef uint fil_h=filters.shape[2]
    cdef uint fil_w=filters.shape[3]

    cdef int fil_mid_h=fil_h//2
    cdef int fil_mid_w=fil_w//2

    cdef uint i, c_in, c_out
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t value

    cdef int y,x,y_off_min, y_off_max, y_off, x_off_min, x_off_max, x_off

    for i in xrange(n_imgs):
        for c_out in xrange(n_channels_out):
            for y in xrange(img_h):
                y_off_min=int_max(-y, -fil_mid_h)
                y_off_max=int_min(img_h-y, fil_mid_h+1)
                for x in xrange(img_w):
                    x_off_min=int_max(-x, -fil_mid_w)
                    x_off_max=int_min(img_w-x, fil_mid_w+1)
                    value=0.0
                    for y_off in xrange(y_off_min, y_off_max):
                        for x_off in xrange(x_off_min, x_off_max):
                            img_y=<uint> (y+y_off)
                            img_x=<uint> (x+x_off)
                            fil_y=<uint> (fil_mid_w+y_off)
                            fil_x=<uint> (fil_mid_h+x_off)
                            for c_in in xrange(n_channels_in):
                                values+=imgs[i,c_in, img_y, img_x]*filters[c_in, c_out, fil_y, fil_x]
                    convout[i, c_out, y, x]=value

def bprop_conv_bc01(np.ndarray[DTYPE_t, ndim=4] imgs, np.ndarray[DTYPE_t, ndim=4] convout_grad, np.ndarray[DTYPE_t, ndim=4] filters, np.ndarray[DTYPE_t, ndim=4] imgs_grad, np.ndarray[DTYPE_t, ndim=4] filters_grad):
    cdef uint n_imgs=convout_grad.shape[0]
    cdef uint img_h=convout_grad.shape[2]
    cdef uint img_w=convout_grad.shape[3]

    cdef uint n_channels_convout= filters.shape[1]
    cdef uint n_channels_imgs= filters.shape[0]

    cdef uint fil_h=filters.shape[2]
    cdef uint fil_w=filters.shape[3]

    cdef int fil_mid_h=fil_h//2
    cdef int fil_mid_w=fil_w//2

    cdef uint i, c_convout, c_imgs
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t convout_grad_value

    cdef int y,x,y_off_min, y_off_max, y_off, x_off_min, x_off_max, x_off
    
    imgs_grad[...]=0
    filters_grad[...]=0
    for i in xrange(n_imgs):
        for c_convout in xrange(n_channels_convout):
            for y in xrange(img_h):
                y_off_min=int_max(-y, -fil_mid_h)
                y_off_max=int_min(img_h-y, fil_mid_h+1)
                for x in xrange(img_w):
                    convout_grad_value=convout_grad[i,c_convout, y, x]
                    x_off_min=int_max(-x, -fil_mid_w)
                    x_off_max=int_min(img_w-x, fil_mid_w+1)
                    for y_off in xrange(y_off_min, y_off_max):
                        for x_off in xrange(x_off_min, x_off_max):
                            img_y=<uint> (y+y_off)
                            img_x=<uint> (x+x_off)
                            fil_y=<uint> (fil_mid_w+y_off)
                            fil_x=<uint> (fil_mid_h+x_off)
                            for c_imgs in xrange(n_channels_imgs):
                                imgs_grad[i, c_imgs, img_y, img_x] +=filters[c_imgs, c_convout, fil_y, fil_x]*convout_grad_value
                                filters_grad[c_imgs, c_convout, fil_y, fil_x] +=imgs[i, c_imgs, img_y, img_x]*convout_grad_value
    filters_grad[...]/=n_imgs

