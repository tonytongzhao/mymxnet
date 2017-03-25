import find_mxnet
import mxnet as mx
from load_data import load
from sklearn.cross_validation import train_test_split
import logging
import pdb as pdb
import numpy as np
import matplotlib.pyplot as plt

def get_mlp():
    """
    multi-layer perceptron
    """

    outLabl = mx.sym.Variable('softmax_label')

    data = mx.symbol.Variable('data')

    flat = mx.symbol.Flatten(data=data)

    fc1  = mx.symbol.FullyConnected(data = flat, name='fc1', num_hidden=100)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=30)
    net  = mx.sym.LinearRegressionOutput(data=fc2, label=outLabl, name='linreg1')

    return net

#
# Load data
#
# Create artificial data
X = np.ones((2140, 9216)).reshape((2140, 1, 96, 96))
y = 0.6*np.ones((2140, 30))

#
# Setup iterators
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
trainIter = mx.io.NDArrayIter(data = X_train, label = y_train, batch_size = 64)
valIter   = mx.io.NDArrayIter(data = X_test , label = y_test , batch_size = 64)

#
# Multidevice kvstore setup and logging
#
kv = mx.kvstore.create('local')
head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

#
# Get model and train
#
net = get_mlp()

model = mx.model.FeedForward(
        ctx                = mx.gpu(),
        symbol             = net,
        num_epoch          = 15,
        learning_rate      = 0.001,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        )
model.fit(X=trainIter, eval_data=valIter, batch_end_callback=mx.callback.Speedometer(1,50), epoch_end_callback=None, eval_metric='rmse')

#
# Prediction
#

valIter.reset()
for prediction in model.predict(valIter):
    print prediction
    pdb.set_trace()
