import numpy as np
import mxnet as mx
import random
import sys
import logging
import argparse
import collections
import model,os
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

igi=0
igv=0.0
def RMSE(label, pred):
    ret=0.0
    n=0.0
    pred=pred.flatten()
    for i in xrange(len(label)):
        ret+=(label[i]-pred[i])**2
        n+=1.0
    return np.sqrt((ret)/(n))

def get_data(data, split, batch_size, user2item, item2user, upass, ipass):
    tr=random.sample(data, int(split*len(data)))
    te=list(set(data)-set(tr))

    return (DataIter(tr, batch_size, user2item, item2user, upass, ipass), DataIter(te, batch_size, user2item, item2user, upass, ipass))

def train(args,data_path, data, val, split, network, batch_size, num_epoch, user2item, item2user, upass, ipass, learning_rate, logname):
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
    if not args.val:
        train, val_iter= get_data(data, split, batch_size, user2item, item2user,  upass, ipass)
    else:
        train=DataIter(data, batch_size, user2item, item2user, upass, ipass)
        val_iter=DataIter(val, batch_size, user2item, item2user, upass, ipass)
    if args.log:
        if not os.path.exists(os.path.join('/'.join(data_path.split('/')[:-1]+[args.model]))):
            os.mkdir(os.path.join('/'.join(data_path.split('/')[:-1]+[args.model])))
        logging.basicConfig(filename=os.path.join('/'.join(data_path.split('/')[:-1]+[args.model]), logname+'.log'), level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.info('start with arguments %s', args)
    network.fit(train, eval_data=val_iter, eval_metric=RMSE, optimizer='NAG', optimizer_params={'learning_rate':learning_rate, 'momentum': 0.995, 'lr_scheduler': mx.lr_scheduler.FactorScheduler(batch_size*3, factor=0.95)}, num_epoch=num_epoch, batch_end_callback=mx.callback.Speedometer(batch_size, 500))
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
def convert_data(user, item, score, user_dict, item_dict, user2item, item2user, is_train=True):
    if user not in user_dict:
        user_dict[user]=len(user_dict)
    user=user_dict[user]
    if item not in item_dict:
        item_dict[item]=len(item_dict)
    item=item_dict[item]
    if is_train:
        user2item[user].add(item)
        item2user[item].add(user)
    return (user, item, float(score))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-train', help='Data file', dest='fi', required=True)
    parser.add_argument('-model', help='cdmemnn, colgru, colattnn', default='cdmemnn')
    parser.add_argument('-nhidden', help='num of hidden', dest='num_hidden', default=50)
    parser.add_argument('-nembed', help='num of embedding', dest='num_embed', default=50)
    parser.add_argument('-batch_size', help='batch size', dest='batch_size', default=500)
    parser.add_argument('-nepoch', help='num of epoch', dest='num_epoch', default=100)
    parser.add_argument('-upass', help='num of collaborative users', dest='upass', default=5)
    parser.add_argument('-ipass', help='num of collaborative items', dest='ipass', default=5)
    parser.add_argument('-npass', help='num of collaborative passes', dest='npass', default=1)
    parser.add_argument('-nlayer', help='num of GRU layers', dest='num_layer', default=1)
    parser.add_argument('-eta', help='learning rate', dest='learning_rate', default=0.01)
    parser.add_argument('-dropout', help='dropout', dest='dropout', default=0.2)
    parser.add_argument('-split',dest='split', help='train & test split ratio', default=0.9)
    parser.add_argument('-val', help='validation set', dest='val', default=None)
    parser.add_argument('-log', help='log to file', default=False)
    args=parser.parse_args()

    user2item=collections.defaultdict(set)
    item2user=collections.defaultdict(set)

    user_dict={}
    item_dict={}
    data=[]
    with open(args.fi, 'r') as f:
        for line in f:
            tks=line.strip().split('\t')
            data.append(convert_data(int(tks[0]), int(tks[1]), float(tks[2]), user_dict, item_dict, user2item, item2user, True))
    val=[]
    if args.val:
        with open(args.val, 'r') as f:
            for line in f:
                tks=line.strip().split('\t')
                if int(tks[0]) not in user_dict or int(tks[1]) not in item_dict:
                    igi+=1
                    igv+=(float(tks[2])-3.0)**2
                    continue
                val.append(convert_data(int(tks[0]), int(tks[1]), float(tks[2]), user_dict, item_dict, user2item, item2user, False))
    print '#Users, ',len(user_dict)
    print '#Items, ',len(item_dict)
    print '#Ratings, ', len(data)
    print 'Ingnored RMSE, ', np.sqrt(igv/igi)
    num_hidden=int(args.num_hidden)
    batch_size=int(args.batch_size)
    num_epoch=int(args.num_epoch)
    learning_rate=float(args.learning_rate)
    num_embed=int(args.num_embed)
    num_layer=int(args.num_layer)
    upass=int(args.upass)
    ipass=int(args.ipass)
    npass=int(args.npass)
    split=float(args.split)
    if args.model=='cdmemnn':
        net=model.get_cdmemnn(batch_size, num_embed, num_hidden, num_layer, len(user_dict), len(item_dict), upass, ipass, npass, float(args.dropout))
    elif args.model=='colgru':
        net=model.get_colgru(batch_size, num_embed, num_hidden, num_layer, len(user_dict), len(item_dict), upass, ipass, npass, float(args.dropout))
    elif args.model=='colattnn':
        net=model.get_c_attention_nn(batch_size, num_embed, num_hidden, num_layer, len(user_dict), len(item_dict), upass, ipass, npass, float(args.dropout))
    elif args.model=='colbilstm':
        net=model.get_colbilstm(batch_size, num_embed, num_hidden, num_layer, len(user_dict), len(item_dict), upass, ipass, npass, float(args.dropout))
    else:
        raise NameError('No matched model, only support cdmemnn, colattnn, colgru')
    logname='upass_'+str(args.upass)+'_ipass_'+str(args.ipass)+"_npass_"+str(args.npass)+"_embed_"+str(args.num_embed)+'_hidden_'+str(args.num_hidden)+"_eta_"+str(args.learning_rate)+"_batch_size_"+str(args.batch_size)
    train(args, args.fi,data, val, split, net, batch_size, num_epoch, user2item, item2user, upass, ipass, learning_rate, logname)

















