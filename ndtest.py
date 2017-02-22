import mxnet as mx
import numpy as np
a=mx.nd.array(np.arange(10).reshape(2,5))
print a.asnumpy()

a=mx.sym.Variable('a')
b=mx.sym.Variable('b')

c=a+b
print (a,b,c)


net=mx.sym.Variable('data')
net=mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net=mx.sym.Activation(data=net, name='relu1', act_type='relu')
net=mx.sym.FullyConnected(data=net, name='fc2', num_hidden=64)
net=mx.sym.Activation(data=net, name='relu2', act_type='relu')
net=mx.sym.FullyConnected(data=net, name='fc3', num_hidden=16)
net=mx.sym.SoftmaxOutput(data=net,name='out')

mx.viz.plot_network(net, shape={'data':(100,200)})
