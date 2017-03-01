import mxnet as mx
import numpy as np
import os
import logging

class NDArraySoftmax(mx.operator.NDArrayOP):
    def __init__(self):
        supper(NDArraySoftmax, self).__init__(False)
        self.fwd_kernel=None
        self.bwd_kernel=None

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['ouput']

    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        label_shape=(in_shape[0][0],)
        output_shape=in_shape[0]

        return [data_shape,label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x=in_data[0]
        y=out_data[0]
        if self.fwd_kernel is None:
            self.fwd_kernel=mx.rtc['softmax', [('x',x)],[('y',y)]]
        self.fwd_kernel.push(([x],[y],(1,1,1), (x.shape[0],1,1)))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l=in_data[1]
        y=out_data[0]
        dx=in_grad[0]
        if self.bwd_kernel==None:
            self.bwd_kernel=mx.rtc('softmax_grad', [('y',y),('l',l)], [('dx',dx)])
        self.bwd_kernel.push([y,l],[dx], (y.shape[0],1,1), (y.shape[1],1,1))
