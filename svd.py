import numpy as np
import mxnet as mx
import random
import sys
import logging
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
    def __init__(self, data, batch_size):
        super(DataIter, self).__init__()
        self.batch_size=batch_size

        self.data=data
        
        self.provide_data=[('user',(self.batch_size,)), ('item', (self.batch_size,))]
        self.provide_label = [('score', (self.batch_size, ))]

    def __iter__(self):
        for k in xrange(len(self.data)/self.batch_size):
            users=[]
            items=[]
            scores=[]
            #Generate each batch data and yield the result
            for i in xrange(self.batch_size):
                j=k*self.batch_size+i
                user, item, score=self.data[j]
                users.append(user)
                items.append(item)
                scores.append(score)
            data_all=[mx.nd.array(users), mx.nd.array(items)]
            label_all=[mx.nd.array(scores)]
            data_names=['user', 'item']
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

def get_data(data,split, n, batch_size):
    tr=random.sample(data, int(n*split))
    te=list(set(data)-set(tr))
    return (DataIter(tr, batch_size), DataIter(te, batch_size))

def train(data, network, split, n, batch_size, num_epoch, learning_rate):
  #  model=mx.mod.Module(network)
    model=mx.model.FeedForward(ctx=mx.gpu(),symbol=network, num_epoch=num_epoch,learning_rate=learning_rate, wd=0.0001, momentum=0.9, initializer=mx.init.Xavier(factor_type="in",magnitude=2.34))
    train, test= get_data(data, split, n,  batch_size)
#    model.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
 #   model.init_params()
    logging.basicConfig(filename='mf.log', level=logging.DEBUG)
    model.fit(train,eval_data=test, eval_metric=RMSE, batch_end_callback=mx.callback.Speedometer(batch_size, 20000/batch_size))

def mf(max_user, max_item, num_hidden):
    u=mx.sym.Variable('user')
    i=mx.sym.Variable('item')
    score = mx.sym.Variable('score')
    user=mx.sym.Embedding(data=u, input_dim=max_user, output_dim= num_hidden, name='user_embed')
    item=mx.sym.Embedding(data=i, input_dim=max_item, output_dim= num_hidden, name='item_embed')
    user_bias=mx.sym.Embedding(data=u, input_dim=max_user, name='embed_user_bias', output_dim=1)
    item_bias=mx.sym.Embedding(data=i, input_dim=max_item, name='embed_item_bias', output_dim=1)

    pred=user*item
    pred=mx.sym.sum_axis(data=pred, axis=1)
    pred=mx.sym.Flatten(data=pred)
    pred=pred+mx.sym.Flatten(data=user_bias)+mx.sym.Flatten(data=item_bias)
    pred=mx.sym.LinearRegressionOutput(data=pred, label = score)
    return pred

def convert_data(user, item, score, user_dict, item_dict):
    if user not in user_dict:
        user_dict[user]=len(user_dict)
    user=user_dict[user]
    if item not in item_dict:
        item_dict[item]=len(item_dict)
    item=item_dict[item]
    return (user, item, float(score))

user_dict={}
item_dict={}
data_file=sys.argv[1]
data=[]
n=0
with open(data_file, 'r') as f:
    for line in f:
        tks=line.strip().split('\t')
        n+=1
        data.append(convert_data(int(tks[0]), int(tks[1]), float(tks[2]), user_dict, item_dict))
print '#User, ', len(user_dict)
print '#Item, ', len(item_dict)
print '#Rating, ', n
num_hidden=300
batch_size=50
num_epoch=2000
learning_rate=0.01
net=mf(len(user_dict), len(item_dict), num_hidden)
train(data, net, 0.9, n, batch_size, num_epoch, learning_rate)

















