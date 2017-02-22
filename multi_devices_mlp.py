import mxnet as mx
import numpy as np
import logging

class mymlp:
    def __init__(self, classes, fc1_hd, fc2_hd, act, output):
        self.fc1_hd=fc1_hd
        self.fc2_hd=fc2_hd
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

def train(network, data_shape, data, devs, devs_power):
    batch_size=float(data_shape[0])
    print batch_size
    workloads=[int(round(batch_size/sum(devs_power)*p)) for p in devs_power]
    print 'workload partition', zip(devs, workloads)
    
    exs=[network.simple_bind(ctx=d, data=tuple([p]+data_shape[1:])) for d,p in zip(devs, workloads)]

    args=[dict(zip(network.list_arguments(), ex.arg_arrays)) for ex in exs]

    for name in args[0]:
        arr=args[0][name]
        arr[:]=mx.random.uniform(-0.1,0.1,arr.shape)

    learning_rate=0.1
    acc=0
    for i in xrange(50):
        #broadcast weight from dev 0 to all others
        for j in xrange(1,len(devs)):
            for name, src, dst in zip(network.list_arguments(), exs[0].arg_arrays, exs[j].arg_arrays):
                src.copyto(dst)
        #get data
        x, y=data()
        for j in xrange(len(devs)):
            idx=range(sum(workloads[:j]), sum(workloads[:j+1]))
            args[j]['data'][:]=x[idx,:].reshape(args[j]['data'].shape)
            args[j]['Softmax_label'][:]=y[idx].reshape(args[j]['Softmax_label'].shape)
            #forward and backward
            exs[j].forward(is_train=True)
            exs[j].backward()
            # sum over/collect all gradient to dev 0
            if j:
                for name, src, dst in zip(network.list_arguments(), exs[j].grad_arrays, exs[0].grad_arrays):
                    dst+=src.as_in_context(dst.context)

        for weight, grad in zip(exs[0].arg_arrays, exs[0].grad_arrays):
            weight[:]-=learning_rate*(grad/batch_size)

        #monitor

        if i%10==0:
            pred=np.concatenate([mx.nd.argmax_channel(ex.outputs[0]).asnumpy() for ex in exs])
            acc=(pred==y).sum()/batch_size
            print 'iteration %d, acc %f' %(i, acc)
    return acc

'''
num_classes=10
num_features=100
toy_data=ToyData(num_classes, num_features)
#x,y=data.get(50000)
batch_size=1000
net=mymlp(num_classes,128,64,'relu','Softmax').construct()
acc=train(net, [batch_size,num_features], lambda: toy_data.get(batch_size), [mx.cpu(), mx.gpu()], [2,2])
assert acc>0.95, 'Low accuracy'

'''


