import mxnet as mx
import numpy as np
import sys
import os
from multi_devices_mlp import ToyData
class SparseLinear(mx.operator.CustomOp):
    def __init__(self, is_data, sparse_reg):
        self.sparse_reg=float(sparse_reg)
        #self.weight=weight
        self.is_data=int(is_data)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], mx.nd.broadcast_mul(in_data[0], in_data[1]))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self.is_data:
            in_grad[0][:]=mx.nd.zeros(in_data[0].shape)
        else:
            in_grad[0][:]=(in_data[1]*mx.nd.sum_axis(out_grad[0],axis=0)).broadcast_to(in_data[0].shape)
        in_grad[1][:]=mx.nd.sum_axis(in_data[0], axis=0)*mx.nd.sum_axis(out_grad[0],axis=0)+((in_data[1]>0)*2-1)*self.sparse_reg
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
        num_class=10
        num_feature=50
        batch_size=100
        toy_data=ToyData(num_class, num_feature)
        
        data=mx.sym.Variable('data')
        sl_weight=mx.sym.Variable('sl_weight')
        data=mx.sym.Custom(data=data, weight=sl_weight, is_data=1, sparse_reg=0.01, name='sl', op_type='sparse_linear')

        #first full c
        fc1=mx.sym.FullyConnected(data=data,num_hidden=500)
        tanh3=mx.sym.Activation(fc1, act_type='tanh')

        #second full c
        fc2=mx.sym.FullyConnected(tanh3, num_hidden=num_class)
    
        #wlr=mx.sym.Custom(data=slr, pos_grad_scale=pos, neg_grad_scale=neg, name='wlr', op_type='weighted_logistic_regression')
        wlr=mx.sym.SoftmaxOutput(data=fc2, name='wlr')

        input_shape={'data':(batch_size, num_feature),  'wlr_label':(batch_size,),'sl_weight': (1,num_feature)}

        executor=wlr.simple_bind(ctx=mx.gpu(), grad_req='write',**input_shape )

        args=dict(zip(wlr.list_arguments(), executor.arg_arrays))
        for r in executor.arg_arrays:
            r[:]=np.random.randn(*r.shape)*0.02
        for epoch in xrange(2):
            x,y=toy_data.get(batch_size)
            executor.arg_dict['data'][:]=mx.nd.array(x,ctx=mx.gpu())
            executor.arg_dict['wlr_label'][:]=mx.nd.array(y,ctx=mx.gpu()).reshape(args['wlr_label'].shape)
            #executor.arg_dict['sl_weight'][:]=mx.nd.array(np.random.rand(1,num_feature),ctx=mx.gpu())

            executor.forward(is_train=True)
            executor.backward()    
            print 'arguments', wlr.list_arguments()
            #print 'arg_arrays', [x for x in executor.arg_arrays]
            #print 'grad_arrays', [x.asnumpy() for x in executor.grad_arrays]
            for pname, w, grad in zip(wlr.list_arguments(), executor.arg_arrays, executor.grad_arrays):
                if pname in ['data', 'wlr_label']:
                    continue
                w[:]-=grad*0.05/batch_size

            if epoch%100==0:
                pred=mx.nd.argmax_channel(executor.outputs[0]).asnumpy()
                print 'epoch %d, acc %f'%(epoch, (pred==y.flatten()).sum()/(batch_size+0.0))
                print 'feature selection\n', args['sl_weight'].asnumpy()

        
