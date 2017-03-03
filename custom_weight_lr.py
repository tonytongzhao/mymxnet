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
        print '\nforward data end'
        self.assign(out_data[0], req[0], mx.nd.array(self.W*in_data[0].asnumpy()))
        print 'forward outdata[0]'
        print out_data[0].asnumpy()
        print 'slr forward end'
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        print 'backward out_data[0]\n', out_data[0].asnumpy()
        if not out_grad[0]:
            print 'No out_grad'
        else:
            print 'backward out_grad', out_grad[0].asnumpy()
            print 'out_grad_len', len(out_grad)

@mx.operator.register('sparse_linear')

class SparseLinearProp(mx.operator.CustomOpProp):
    def __init__(self, dim, sparse_reg):
        self.sparse_reg=sparse_reg
        self.dim=dim
        super(SparseLinearProp, self).__init__( need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        shape=in_shape[0]
        return [shape], [shape]

    def create_operator(self, ctx, shape, dtypes):
        return SparseLinear(self.dim, self.sparse_reg)

        



if __name__=='__main__':
    if True:    
        m,n=2,5
        pos,neg=1.0, 0.1
        dim=(m*2,n)
        data=mx.sym.Variable('data')
        data=mx.sym.Custom(data=data,dim=dim, sparse_reg=0.01, name='slr', op_type='sparse_linear')
        slr=mx.sym.FullyConnected(data=data, num_hidden=5, name='fc1')
        #data=mx.sym.FullyConnected(data=data, num_hidden=5)
        #wlr=mx.sym.Custom(data=slr, pos_grad_scale=pos, neg_grad_scale=neg, name='wlr', op_type='weighted_logistic_regression')
        slr=mx.sym.sum_axis(data=slr, axis=1, name='sum1')
        slr=mx.sym.Flatten(data=slr, name='flatten1')
        wlr=mx.sym.LinearRegressionOutput(data=slr, name='wlr')
        mod_test=mx.mod.Module(symbol=wlr, data_names=['data', 'wlr_label'], context=mx.gpu())
        mod_test.bind(data_shapes=[('data', (2*m,5 )), ('wlr_label', (2*m,1))])
        mod_test.init_params(initializer=mx.init.Xavier(factor_type='in', magnitude=1))
        mod_test.init_optimizer(optimizer='adam', kvstore=None, optimizer_params={'learning_rate':1E-3, 'wd':1E-4})
        batch_data=mx.io.DataBatch(data=[mx.nd.array(np.arange(20).reshape((4,5))),mx.nd.array(np.vstack([np.ones((m,1)), np.zeros((m,1))]))], label=['data', 'wlr_label'])
        mod_forward=mod_test.forward(data_batch=batch_data, is_train=True)
        print 'output[0] is ', mod_test.get_outputs()[0].asnumpy()
        print mod_test.get_params()
        mod_test.backward()
        mod_test.update()
        print mod_forward.grad_dict.keys()
        mid=wlr.get_internals()
        print mid.list_outputs()

        

