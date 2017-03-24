import mxnet as mx
import os, urllib, zipfile
import mylstm
import numpy as np
from variable_bucket import BucketFlexIter 
from bi_lstm import bi_lstm_unroll
import argparse
import random, string
import jason
from utils import read_content, build_vocab, text2id



def Perplexity(label, pred):
    loss=0.
    for i in range(pred.shape[0]):
        loss+=-np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss/label.size)

def MakeRevertVocab(vocab):
	dic={}
	for k, v in vocab.items():
		dic[v]=k
	return dic


def train(path, df, nhidden, nembed, batch_size, epoch, model, nlayer, eta, dropout, split):
    assert model in ['lstm', 'bilstm', 'gru']
    buckets=[10,50,100,150,200]
     
    





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
    










