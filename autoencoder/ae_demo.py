import model
import autoencoder
import mxnet as mx
import logging 

import numpy as np

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    aemodel=autoencoder.AutoEncoderModel(mx.gpu(), [784,500,500,2000,10], pt_dropout=0.2, internal_act='relu', output_act='relu')

    X
