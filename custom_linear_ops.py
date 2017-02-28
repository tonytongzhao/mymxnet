import mxnet as mx
import os
import numpy as np

class l1_sparse(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        
