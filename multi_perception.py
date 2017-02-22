import mxnet as mx
import logging
import numpy as np
from data_iter import SyntheticData


net=mx.sym.Variable('data')
net=mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net=mx.sym.Activation(net, name='relu1', act_type='relu')
net=mx.sym.FullyConnected(net, name='fc2', num_hidden=64)
net=mx.sym.SoftmaxOutput(net, name='softmax')

data=SyntheticData(10,128)

mod= mx.mod.Module(symbol=net, context=mx.gpu(), data_names=['data'], label_names=['softmax_label'])

logging.basicConfig(level=logging.INFO)
batch_size=32
mod.fit(data.get_iter(batch_size), eval_data=data.get_iter(batch_size), optimizer='sgd',optimizer_params={'learning_rate':0.1},eval_metric='acc', num_epoch=5)



