import mxnet as mx
import numpy as np

class SparseLinear(mx.operator.CustomOp):
    def __init__(self, dim, random_seed=54321):
        super(SparseLinear, self).__init__()
        self.dim=int(dim)
        self.W=np.random.RandomState(seed=random_seed).normal(size=(1,self.dim))

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], mx.nd.array(self.W*in_data[0]))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        print 'out_grad\n'
        print out_grad[0].asnumpy()
@mx.operator.register('splinear')

class SparseLinearProp(mx.operator.CustomOpProp):
    def __init__(self, dim):
        self.dim=int(dim)
        super(SparseLinearProp, self).__init__(need_top_grad=True)
    def list_arguments(self):
        return ['data', 'w']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        batch_size=in_shape[0][0]
        return [in_shape[0], in_shape[0]], [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return SparseLinear(self.dim)


