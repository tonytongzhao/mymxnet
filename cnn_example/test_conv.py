import numpy as np
import scipy.ndimage.filters

import pyximport

pyximport.install()

from conv import conv_bc01

def test_conv():
    img=np.random.randn(4,4)
    imgs=img[np.newaxis, np.newaxis, ...]
    print imgs


if __name__=='__main__':
    test_conv()
