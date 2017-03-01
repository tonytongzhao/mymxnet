import mxnet as mx
import numpy as np
import logging

class NumpySoftmax(mx.operator.NumpyOp):
    def __init__(self):
        super(NumpySoftmax, self).__init__(False)
    
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']
    
    def infer_shape(self, in_shape):
        #in_shape= [data_shape]
        #data_shape=[batch_size, ]
        data_shape=in_shape[0]
        label_shape=(in_shape[0][0],)
        output_shape=in_shape[0]
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x=in_data[0]
        y=out_data[0]
        y[:]=np.exp(x-x.max(axis=1).reshape((x.shape[0],1)))
        y/=y.sum(axis=1).reshape((x.shape[0],1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l=in_data[1]
        l=l.reshape((l.size,1)).astype(np.int)
        y=out_data[0]
        dx=in_grad[0]
        dx[:]=y
        dx[np.arange(l.shape[0]),l]-=1.0

