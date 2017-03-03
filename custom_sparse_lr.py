import mxnet as mx
import numpy as np
import sys
import os

class SparseLinear(mx.operator.CustomOp):
    def __init__(self, is_data, sparse_reg):
        self.sparse_reg=float(sparse_reg)
        #self.weight=weight
        self.is_data=int(is_data)
        self.dim=dim

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], mx.nd.broadcast_mul(in_data[0], in_data[1]))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self.is_data:
            in_grad[0][:]=mx.nd.zeros(in_data[0].shape)
        else:
            in_grad[0][:]=(in_data[1]*mx.nd.sum_axis(out_grad[0],axis=0)).broadcast_to(in_data[0].shape)
        in_grad[1][:]=mx.nd.sum_axis(in_data[0], axis=0)*mx.nd.sum_axis(out_grad[0],axis=0)/in_data[0].shape[0]+((in_data[1]>0)*2-1)*self.sparse_reg
@mx.operator.register('sparse_linear')

class SparseLinearProp(mx.operator.CustomOpProp):
    def __init__(self, is_data, sparse_reg):
        self.sparse_reg=sparse_reg
        self.is_data=is_data
        super(SparseLinearProp, self).__init__( need_top_grad=True)

    def list_arguments(self):
        return ['data', 'weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        shape=in_shape[0]
        return [shape,in_shape[1]], [shape]

    def create_operator(self, ctx, shape, dtypes):
        return SparseLinear( self.is_data, self.sparse_reg)

        



if __name__=='__main__':
    if True:    
        m,n=2,5
        pos,neg=1.0, 0.1
        dim=(m*2,n)
        data=mx.sym.Variable('data')
        sl_weight=mx.sym.Variable('sl_weight')
        sldata=mx.sym.Custom(data=data, weight=sl_weight, is_data=1, sparse_reg=0.01, name='sl', op_type='sparse_linear')
        slr=mx.sym.FullyConnected(data=sldata, num_hidden=15, name='fc1')
        #data=mx.sym.FullyConnected(data=data, num_hidden=5)
        #wlr=mx.sym.Custom(data=slr, pos_grad_scale=pos, neg_grad_scale=neg, name='wlr', op_type='weighted_logistic_regression')
        slr=mx.sym.sum_axis(data=slr, axis=1, name='sum1')
        slr=mx.sym.Flatten(data=slr, name='flatten1')
        wlr=mx.sym.LinearRegressionOutput(data=slr, name='wlr')

        input_shape={'data':(2*m,n), 'sl_weight': (1,n), 'wlr_label':(2*m,1)}
        executor=wlr.simple_bind(ctx=mx.gpu(), grad_req='write',**input_shape )

        for r in executor.arg_arrays:
            r[:]=np.random.randn(*r.shape)*0.02

        executor.arg_dict['data'][:]=mx.nd.array(np.arange(20).reshape((4,5)),ctx=mx.gpu())
        executor.arg_dict['wlr_label'][:]=mx.nd.array(np.vstack([np.ones((m,1)), np.zeros((m,1))]), ctx=mx.gpu())
        executor.arg_dict['sl_weight'][:]=mx.nd.array(np.random.rand(1,n),ctx=mx.gpu())

        executor.forward()
        executor.backward()    

        print 'arguments', wlr.list_arguments()
        #print 'arg_arrays', [x for x in executor.arg_arrays]
        print 'grad_arrays', [x.asnumpy() for x in executor.grad_arrays]
        
        
