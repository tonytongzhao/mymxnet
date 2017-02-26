import numpy as np
import mxnet as mx
import random
import sys
import logging
import itertools
class Batch:
    def __init__(self, data_names, data, label_names, label, pad=0):
        self.data=data
        self.data_names=data_names
        self.label=label
        self.label_names=label_names

    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    def provide_label(self):
        return [(n,x.shape) for n, x in zip(self.label_names, self.label)]


class DataIter(mx.io.DataIter):
    def __init__(self, data, num_field, batch_size):
        super(DataIter, self).__init__()
        self.batch_size=batch_size
        self.num_field=num_field
        self.data=data
        
        self.provide_data=[('x_%d'%i,(self.batch_size,)) for i in xrange(self.num_field)]
        self.provide_label = [('score', (self.batch_size, ))]

    def __iter__(self):
        for k in xrange(len(self.data)/self.batch_size):
            x=[[] for _ in xrange(self.num_field)]
            scores=[]
            #Generate each batch data and yield the result
            for i in xrange(self.batch_size):
                j=k*self.batch_size+i
                assert num_field+1==len(self.data[j])
                for f in xrange(num_field):
                    x[f].append(self.data[j][f])
                scores.append(self.data[j][-1])
            data_all=[mx.nd.array(i) for i in x]
            label_all=[mx.nd.array(scores)]
            data_names=['x_%d'%i for i in xrange(self.num_field)]
            label_names=['score']

            data_batch=Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        random.shuffle(self.data)

def RMSE(label, pred):
    ret=0.0
    n=0.0
    pred=pred.flatten()
    for i in xrange(len(label)):
        ret+=(label[i]-pred[i])**2
        n+=1.0
    return np.sqrt(ret/n)

def get_data(data, num_field, batch_size):
    return (DataIter(data, num_field, batch_size), DataIter(data,num_field, batch_size))

def train(data,num_field, network, batch_size, num_epoch, learning_rate):
    '''
    network=mx.mod.Module(network, data_names=tuple(['x%d'%(i) for i in xrange(num_field)]),label_names=('score'),context=mx.gpu())
    network.bind(data_shapes=[('x_%d'%i,(batch_size,)) for i in xrange(num_field)], label_shapes=[('score',(batch_size,))], grad_req='write')
    init=mx.init.Xavier(factor_type='in', magnitude=1)
    network.init_params(initializer=init)
    network.init_optimizer(optimizer='adam', kvstore=None, optimizer_params={'learning_rate':1E-3, 'wd':1E-4})
    for i in xrange(num_epoch):
        batch_data=random.sample(data, batch_size)
        x=[[] for _ in xrange(num_field)]
        score=[]
        for d in batch_data:
            for j in xrange(len(d)-1):
                x[j].append(d[j])
            score.append(d[-1])
        network.forward(data_batch=mx.io.DataBatch(data=x,label=[score]), is_train=True)
        outputs=network.get_outputs()
        network.backward()
        logging.info("Iter:%d, Error:%f" %(i,outputs[0].asnumpy().sum()/(batch_size+0.0)))	
        network.update()
    '''
    model=mx.model.FeedForward(ctx=mx.gpu(),symbol=network, num_epoch=num_epoch,learning_rate=learning_rate, wd=0.0001, momentum=0.9, initializer=mx.init.Xavier(factor_type="in",magnitude=2.34))
    train, test= get_data(data,num_field, batch_size)
    logging.basicConfig(level=logging.DEBUG)
    model.fit(train,eval_data=test, eval_metric=RMSE, batch_end_callback=mx.callback.Speedometer(batch_size, 20000/batch_size))



def deepffm(field_dim,num_hidden, num_layer):
    w=[]
    x=[]
    v=[]
    print field_dim
    score=mx.sym.Variable('score')
    num_field=len(field_dim)
    for i in xrange(num_field):
        x.append(mx.sym.Variable('x_%d'%(i)))
    for i in xrange(num_field):
        w.append(mx.sym.Embedding(data=x[i], name='w_f%d'%i, input_dim=field_dim[i], output_dim=1))
        #w[i]=mx.sym.Reshape(data=w[i], shape=())
        v.append(mx.sym.Embedding(data=x[i], name='v_eb_f%d_l%d'%(i,0), input_dim=field_dim[i], output_dim=num_hidden))
    for l in xrange(1,num_layer):
        for f in xrange(num_field):
            v[f]=mx.sym.FullyConnected(data=v[f], name='v_fc_f%d_l%d'%(f,l), num_hidden=num_hidden)
            v[f]=mx.sym.Activation(data=v[f],name='v_act_f%d_l%d'%(f,l), act_type='relu')
            v[f]=mx.sym.Dropout(data=v[f],p=0.2)
    
    #wx=[wi*xi for wi,xi in zip(w,x)]
    #wx=mx.sym.ElementWiseSum(*wx)
    ''' 
    
    #FFM all combine
    vsum=mx.sym.ElementWiseSum(*v)
    
    vsq=[mx.sym.square(i) for i in v]
    vsqsum=mx.sym.ElementWiseSum(*vsq)
    
    res=(vsqsum-vsum)/2.0 
    '''
    res=[v[i]*v[j] for i,j in itertools.combinations(range(num_field),2)]
    res=mx.sym.ElementWiseSum(*res)
    #res=res+wx
    res=mx.sym.Dropout(data=res, p=0.2) 
    #res=vsum
    #res=v[0]*v[1]
    res=mx.sym.sum_axis(data=res, axis=1)
    res=mx.sym.Flatten(data=res)
    res=mx.sym.LinearRegressionOutput(data=res, label=score)
    return res

def convert_data(user, item, score, user_dict, item_dict):
    if user not in user_dict:
        user_dict[user]=len(user_dict)
    user=user_dict[user]
    if item not in item_dict:
        item_dict[item]=len(item_dict)
    item=item_dict[item]
    return (user, item,float(score))

user_dict={}
item_dict={}
data_file=sys.argv[1]
data=[]
with open(data_file, 'r') as f:
    for line in f:
        tks=line.strip().split('\t')
        data.append(convert_data(int(tks[0]), int(tks[1]), float(tks[2]), user_dict, item_dict))
max_user=len(user_dict)
max_item=len(item_dict)
print '#user ',max_user
print '#item ',max_item
num_hidden=50
batch_size=50
num_epoch=200
learning_rate=0.01
num_field=2
num_layer=5
field_dims=[max_user, max_item]
net=deepffm(field_dims, num_hidden, num_layer)
train(data, num_field, net, batch_size, num_epoch, learning_rate)

















