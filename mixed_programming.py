import mxnet as mx
import numpy as np
import logging

class mymlp:
    def __init__(self, classes, fc1_hd, fc2_hd, act, output):
        self.fc1_hd=fc1_hd
        self.fc2_hd=fc1_hd
        self.act=act
        self.output=output
        self.classes=classes
    def construct(self):
        net=mx.sym.Variable('data')
        net=mx.sym.FullyConnected(net, name='fc1', num_hidden=self.fc1_hd)
        net=mx.sym.Activation(net, name=self.act, act_type=self.act)
        net=mx.sym.FullyConnected(net, name='fc2', num_hidden=self.fc2_hd)
        if self.output=='Softmax':
            net=mx.sym.SoftmaxOutput(net, name=self.output)
        self.net=net
        return self.net

class ToyData(object):
    def __init__(self, num_classes, num_features):
        self.num_classes=num_classes
        self.num_features=num_features
        self.mu=np.random.rand(num_classes, num_features)
        self.sigma=np.ones((num_classes, num_features))*0.1
    
    def get(self, num_samples):
        num_cls_samples=num_samples/self.num_classes
        x=np.zeros((num_samples, self.num_features))
        y=np.zeros((num_samples,))
        for i in xrange(self.num_classes):
            cls_samples=np.random.normal(self.mu[i,:], self.sigma[i,:], (num_cls_samples, self.num_features))
            x[i*num_cls_samples:(i+1)*num_cls_samples]=cls_samples 
            y[i*num_cls_samples:(i+1)*num_cls_samples]=i

        return x,y


num_classes=10
num_features=100
#x,y=data.get(50000)
batch_size=100
net=mymlp(num_classes,128,64,'relu','Softmax').construct()
ex=net.simple_bind(ctx=mx.gpu(),data=(batch_size, num_features))
args=dict(zip(net.list_arguments(), ex.arg_arrays))
for name in args:
    print name, args[name].shape, args[name].context

for name in args:
    data=args[name]
    data[:]=mx.random.uniform(-0.1,0.1,data.shape)

learning_rate=0.1
final_acc=0.0

data=ToyData(10,100)
for i in xrange(100):
    x,y=data.get(batch_size)
    args['data'][:]=x
    args['Softmax_label'][:]=y
    ex.forward(is_train=True)
    ex.backward()
    for weight, grad in zip(ex.arg_arrays, ex.grad_arrays):
        weight[:]-=learning_rate*(grad/batch_size)
    if i%10==0:
        acc=(mx.nd.argmax_channel(ex.outputs[0]).asnumpy()==y).sum()
        final_acc=acc
        print 'iteration %d, acc %f' %(i, float(acc)/y.shape[0])

assert final_acc>0.95, 'Low accuracy'




