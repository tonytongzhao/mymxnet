from  multi_devices_mlp import train
from sklearn.datasets import fetch_mldata
import numpy as np
import mxnet as mx
import time
def lenet():
    data=mx.sym.Variable('data')
    #first conv
    conv1=mx.sym.Convolution(data,kernel=(5,5), num_filter=20)
    tanh1=mx.sym.Activation(conv1, act_type='tanh')
    pool1=mx.sym.Pooling(tanh1, pool_type='max', kernel=(2,2), stride=(2,2))

    #second conv
    conv2=mx.sym.Convolution(pool1,kernel=(5,5), num_filter=50)
    tanh2=mx.sym.Activation(conv2, act_type='tanh')
    pool2=mx.sym.Pooling(tanh2, pool_type='max', kernel=(2,2), stride=(2,2))

    #first full c
    flatten=mx.sym.Flatten(pool2)
    fc1=mx.sym.FullyConnected(flatten,num_hidden=500)
    tanh3=mx.sym.Activation(fc1, act_type='tanh')

    #second full c
    fc2=mx.sym.FullyConnected(tanh3, num_hidden=10)
    
    lenet=mx.sym.SoftmaxOutput(fc2, name='Softmax')
    return lenet

class MNIST:
    def __init__(self):
        mnist=fetch_mldata('MNIST original')
        p=np.random.permutation(mnist.data.shape[0])
        self.x=mnist.data[p]
        self.y=mnist.target[p]
        self.pos=0
    def get(self, batch_size):
        p=self.pos
        self.pos+=batch_size
        return self.x[p:self.pos,:], self.y[p:self.pos, :]

mnist=MNIST()
batch_size=1024
shape=[batch_size, 1, 28, 28]
print batch_size
tic=time.time()
acc=train(lenet(), shape, lambda: mnist.get(batch_size), [mx.gpu(),],[1,])
print 'time with single gpu', time.time()-tic
