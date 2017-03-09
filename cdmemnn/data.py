import numpy as np
import mxnet as mx
import random
import sys
import logging
import collections
import model
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
    def __init__(self, data, batch_size, user2item, item2user, upass, ipass):
        super(DataIter, self).__init__()
        self.batch_size=batch_size
        self.upass=upass
        self.ipass=ipass
        self.data=data
        self.user2item=user2item
        self.item2user=item2user
        self.provide_data=[('user',(self.batch_size,)), ('item', (self.batch_size,)), ('grp_u', (self.batch_size,upass)), ('grp_i', (self.batch_size,ipass)),]
        self.provide_label = [('rating', (self.batch_size,)),]

    def __iter__(self):
        for k in xrange(len(self.data)/self.batch_size):
            users=[]
            items=[]
            scores=[]
            grp_u=[]
            grp_i=[]
           #Generate each batch data and yield the result
            for i in xrange(self.batch_size):
                j=k*self.batch_size+i
                user, item, score=self.data[j]
                users.append(user)
                items.append(item)
                scores.append(score)
                ug=list(item2user[item])+[user]*(upass-len(item2user[item])) if upass>len(item2user[item]) else random.sample(item2user[item], upass)
                ig=list(user2item[user])+[item]*(ipass-len(user2item[user])) if ipass>len(user2item[user]) else random.sample(user2item[user], ipass)
                grp_u.append(ug)
                grp_i.append(ig)
                    
            data_all=[mx.nd.array(users), mx.nd.array(items), mx.nd.array(grp_u), mx.nd.array(grp_i)]
            label_all=[mx.nd.array(scores)]
            data_names=['user', 'item', 'grp_u', 'grp_i',]
            label_names=['rating',]

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

def get_data(data, batch_size, user2item, item2user, upass, ipass):
    return (DataIter(data, batch_size, user2item, item2user, upass, ipass), DataIter(data, batch_size, user2item, item2user, upass, ipass))

def train(data, network, batch_size, num_epoch, user2item, item2user, learning_rate, upass, ipass):
    ''' 
    model=mx.model.FeedForward(ctx=mx.gpu(),symbol=network, num_epoch=num_epoch,learning_rate=learning_rate, wd=0.0001, momentum=0.9, initializer=mx.init.Normal(sigma=0.01))
    train, test= get_data(data, batch_size, user2item, item2user,  upass, ipass)
    logging.basicConfig(level=logging.DEBUG)
    model.fit(train,eval_data=test, eval_metric=RMSE, batch_end_callback=mx.callback.Speedometer(batch_size, 20000/batch_size))
    '''
    network=mx.mod.Module(network, data_names=('user', 'item', 'grp_u', 'grp_i',), label_names=('rating',), context=mx.gpu())
    #network.bind(data_shapes=[('user', (batch_size,)), ('item', (batch_size,)), ('grp_u', (batch_size,)), ('grp_i', (batch_size,)),], label_shapes=[('rating', (batch_size,)),])

    #init=mx.init.Xavier(factor_type='in', magnitude=1)
    #network.init_params(initializer=init)
    #network.init_optimizer(optimizer='adam', kvstore=None, optimizer_params={'learning_rate':1E-3, 'wd':1E-4}) 
    train, test= get_data(data, batch_size, user2item, item2user,  upass, ipass)
    logging.basicConfig(level=logging.DEBUG)
    network.fit(train, eval_data=test, eval_metric=RMSE, num_epoch=num_epoch)
    '''
    for i in xrange(num_epoch):
        batch_data=random.sample(data, batch_size)
        users=[]
        items=[]
        scores=[]
        grp_u=[]
        grp_i=[]
        #Generate each batch data and yield the result
        for x in batch_data:
            user, item, score=x
            users.append(user)
            items.append(item)
            scores.append(score)
            ug=list(item2user[item])+[user]*(upass-len(item2user[item])) if upass>len(item2user[item]) else random.sample(item2user[item], upass)                       ig=list(user2item[user])+[item]*(ipass-len(user2item[user])) if ipass>len(user2item[user]) else random.sample(user2item[user], ipass)
            grp_u.append(ug)
            grp_i.append(ig)
        network.forward(data_batch=mx.io.D)
    '''
def convert_data(user, item, score, user_dict, item_dict, user2item, item2user):
    if user not in user_dict:
        user_dict[user]=len(user_dict)
    user=user_dict[user]
    if item not in item_dict:
        item_dict[item]=len(item_dict)
    item=item_dict[item]
    user2item[user].add(item)
    item2user[item].add(user)
    return (user, item, float(score))

user2item=collections.defaultdict(set)
item2user=collections.defaultdict(set)

user_dict={}
item_dict={}
data_file=sys.argv[1]
data=[]
with open(data_file, 'r') as f:
    for line in f:
        tks=line.strip().split('\t')
        data.append(convert_data(int(tks[0]), int(tks[1]), float(tks[2]), user_dict, item_dict, user2item, item2user))
print len(user_dict), len(item_dict)
num_hidden=100
batch_size=50
num_epoch=2000
learning_rate=0.01
num_embed=150
num_layer=1
upass=15
ipass=15
npass=10

net=model.get_cdnn(batch_size, num_embed, num_hidden, num_layer, len(user_dict), len(item_dict), upass, ipass, npass)
train(data, net, batch_size, num_epoch, learning_rate, user2item, item2user, upass, ipass)

















