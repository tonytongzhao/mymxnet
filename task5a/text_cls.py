import mxnet as mx
import os, urllib, zipfile
import lstm,gru
import numpy as np
from variable_bucket import BucketFlexIter 
from bilstm import bi_lstm_unroll
import argparse
import random, string
from utils import read_content, load_data, text2id
import numpy as np
import logging
def accuracy(label, pred):
    return np.sum(label*np.round(pred))/(0.0+np.sum(label))

def Perplexity(label, pred):
    loss=0.
    for i in range(pred.shape[0]):
        loss+=-np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss/label.size)

def get_data_iter(ins, labels, nlabels, batch_size, init_states, buckets, split):
    num_ins=len(ins)
    tr=random.sample(range(num_ins), int(split)*num_ins)
    te=list(set(range(num_ins))-set(tr))
    return BucketFlexIter(ins[tr], labels[tr], nlabels, batch_size, init_states, buckets), BucketFlexIter(ins[te], labels[te], nlabels, batch_size, init_states, buckets) 

def train(path, df, nhidden, nembed, batch_size, nepoch, model, nlayer, eta, dropout, split):
    data=read_content(os.path.join(path, df))
    ins, labels, vocab, label_dict= load_data(data)
    print '#ins', len(ins)
    print '#labels', len(label_dict)
    print '#words', len(vocab)
    contexts=[mx.context.gpu(i) for i in xrange(1)]   
    nwords=len(vocab)
    nlabels=len(label_dict)
    buckets=[10,50,100,150,200]
    logging.basicConfig(level=logging.DEBUG)
    assert model in ['lstm', 'bilstm', 'gru']
    if model =='lstm':
        init_c = [('l%d_init_c'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_h = [('l%d_init_h'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_states = init_c + init_h
	state_names=[x[0] for x in init_states]
	tr_data, val_data=get_data_iter(ins, labels, nlabels, batch_size, init_states, buckets, split)
	def lstm_gen(seq_len):
            sym=lstm.lstm_unroll(nlayer, seq_len, nwords, nhidden, nembed, nlabels, dropout)
	    data_names=['data']+state_names
	    label_names=['label']
	    return sym, data_names, label_names
	if len(buckets) == 1:
	    mod = mx.mod.Module(*lstm_gen(buckets[0]), context=contexts)
        else:
	    mod = mx.mod.BucketingModule(lstm_gen, default_bucket_key=tr_data.default_bucket_key, context=contexts) 
        mod.fit(tr_data, eval_data=val_data, num_epoch=nepoch,eval_metric=mx.metric.np(accuracy),batch_end_callback=mx.callback.Speedometer(batch_size, 50),initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), optimizer='sgd', optimizer_params={'learning_rate':0.01, 'momentum': 0.9, 'wd': 0.00001})
                                                                    




if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-path', help='data path', dest='path', required=True)
    parser.add_argument('-file', help='data file', dest='fi', required=True)
    parser.add_argument('-nhidden', help='num of hidden', dest='num_hidden', default=50)
    parser.add_argument('-nembed', help='num of embedding', dest='num_embed', default=50)
    parser.add_argument('-batch_size', help='batch size', dest='batch_size', default=100)
    parser.add_argument('-nepoch', help='num of epoch', dest='num_epoch', default=200)
    parser.add_argument('-nlayer', help='num of GRU layers', dest='num_layer', default=1)
    parser.add_argument('-eta', help='learning rate', dest='learning_rate', default=0.005)
    parser.add_argument('-dropout', help='dropout', dest='dropout', default=0.2)
    parser.add_argument('-split',dest='split', help='train & validation split ratio', default=0.9)
    parser.add_argument('-model', dest='model', help='model module: lstm, bilstm, gru', required=True)
    args=parser.parse_args()
    path=args.path
    df=args.fi
    nhidden=int(args.num_hidden)
    nembed=int(args.num_embed)
    batch_size=int(args.batch_size)
    epoch=int(args.num_epoch)
    model=args.model
    nlayer=int(args.num_layer)
    eta=float(args.learning_rate)
    dropout=float(args.dropout)
    split=float(args.split)

    train(path, df, nhidden, nembed, batch_size, epoch, model, nlayer, eta, dropout, split)
    










