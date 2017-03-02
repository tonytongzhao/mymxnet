import mxnet as mx
import numpy as np
import sys
import os

class WeightedLogisticRegression(mx.operator.CustomOp):
    def __init__(self, pos_grad_scale, neg_grad_scale):
        self.pos_grad_scale=float(pos_grad_scale)
        self.neg_grad_scale=float(neg_grad_scale)

    def forward(self, is_train, req, in_data, out_data, aux):
        print 'wlr forward'
        self.assign(out_data[0], req[0], mx.nd.divide(1, (1+mx.nd.exp(-in_data[0]))))
    
    def backward(self, req, out_grad, in_data,out_data, in_grad, aux):
        print 'wlr backward'
        in_grad[0][:]=((out_data[0]-1)*in_data[1]*self.pos_grad_scale + out_data[0]*(1-in_data[1])*self.neg_grad_scale)/out_data[0].shape[1]

@mx.operator.register('weighted_logistic_regression')

class WeightedLogisticRegressionProp(mx.operator.CustomOpProp):
    def __init__(self, pos_grad_scale, neg_grad_scale):
        self.pos_grad_scale=pos_grad_scale
        self.neg_grad_scale=neg_grad_scale
        super(WeightedLogisticRegressionProp, self).__init__(False)

    def list_arguments(self):
        return ['data', 'label']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        shape=in_shape[0]
        return [shape, shape], [shape]

    def create_operator(self, ctx, shape, dtypes):
        return WeightedLogisticRegression(self.pos_grad_scale, self.neg_grad_scale)


class SparseLinear(mx.operator.CustomOp):
    def __init__(self, dim, sparse_reg):
        self.sparse_reg=sparse_reg
        self.dim=dim
        rs=np.random.RandomState(seed=12345)
        self.W=rs.normal(size=(4,5))
        print 'self.W', self.W
        print '\n'

    def forward(self, is_train, req, in_data, out_data, aux):
        print 'forward data', in_data[0].asnumpy()
        print '\nforward data end\n'
        self.assign(out_data[0], req[0], mx.nd.array(self.W*in_data[0].asnumpy()))
        print 'forward outdata[0]\n'
        print out_data[0].asnumpy()

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        print 'backward out_data[0]\n', out_data[0].asnumpy()
        print '\n'
        if not out_grad:
            print 'No out_grad'
        else:
            print '\nbackward out_grad', out_grad[0].asnumpy()
            print 'out_grad_len', len(out_grad)

@mx.operator.register('sparse_linear')

class SparseLinearProp(mx.operator.CustomOpProp):
    def __init__(self, dim, sparse_reg):
        self.sparse_reg=sparse_reg
        self.dim=dim
        super(SparseLinearProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data', 'w', 'b']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        shape=in_shape[0]
        return [shape, shape, shape], [shape]

    def create_operator(self, ctx, shape, dtypes):
        return SparseLinear(self.dim, self.sparse_reg)

        



if __name__=='__main__':
    if True:    
        m,n=2,5
        pos,neg=1.0, 0.1
        dim=(m*2, n)
        data=mx.sym.Variable('data')
        #slr=mx.sym.FullyConnected(data=data, num_hidden=10)
        data=mx.sym.Custom(data=data,dim=dim, sparse_reg=0.01, name='slr', op_type='sparse_linear')
        #data=mx.sym.FullyConnected(data=data, num_hidden=5)
        wlr=mx.sym.Custom(data=data, pos_grad_scale=pos, neg_grad_scale=neg, name='wlr', op_type='weighted_logistic_regression')
        

        
        exe_wlr=wlr.simple_bind(ctx=mx.gpu(0), data=(2*m,n))
        exe_wlr.arg_dict['data'][:]=np.arange(2*m*n).reshape([2*m,n])
        exe_wlr.arg_dict['wlr_label'][:]=np.vstack([np.ones([m,n]), np.zeros([m,n])])
        exe_wlr.forward(is_train=True)
        exe_wlr.backward()
        exe_wlr.update() 

        

        '''
        print 'wlr output'
        print exe_wlr.outputs[0].asnumpy()
        
        if True:
            print wlr.list_arguments()
            print 'args'
            print len(exe_wlr.arg_arrays)
            print [x[0].asnumpy() for x in  exe_wlr.arg_arrays]
            print [x[1].asnumpy() for x in  exe_wlr.arg_arrays]
            print [x[2].asnumpy() for x in  exe_wlr.arg_arrays]
            print [x[3].asnumpy() for x in  exe_wlr.arg_arrays]
                
            print 'grads'
            print len(exe_wlr.grad_arrays)
            print [x[0].asnumpy() for x in exe_wlr.grad_arrays]
            print [x[1].asnumpy() for x in exe_wlr.grad_arrays]
            print [x[2].asnumpy() for x in exe_wlr.grad_arrays]
            print [x[3].asnumpy() for x in exe_wlr.grad_arrays]
        '''

