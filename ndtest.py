import mxnet as mx
import numpy as np
a=mx.nd.array(np.arange(10).reshape(2,5))
print a.asnumpy()
