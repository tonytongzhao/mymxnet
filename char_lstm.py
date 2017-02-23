import mxnet as mx
import os, urllib, zipfile
import mylstm
import numpy as np
import bucket_io
#if not os.path.exists('char_lstm.zip'):
#	urllib.urlretrieve('http://data.mxnet.io/data/char_lstm.zip', 'char_lstm.zip')
#with zipfile.ZipFile('char_lstm.zip', 'r') as f:
#	f.extractall('./')


def read_content(path):
	with open(path) as ins:
		return ins.read()

def build_vocab(path):
	content=list(read_content(path))
	idx=1
	vocab={}
	for w in content:
		if len(w) and w not in vocab:
			vocab[w]=idx
			idx+=1
	return vocab

def text2id(sentence, vocab):
    words=list(sentence)
    return [vocab[w] for w in words if len(w)]


def Perplexity(label, pred):
    loss=0.
    for i in range(pred.shape[0]):
        loss+=-np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss/label.size)
vocab=build_vocab('./mldata/obama.txt')

seq_len=129
num_embed=256
num_lstm_layer=3
num_hidden=512


symbol=mylstm.lstm_unroll(num_lstm_layer, seq_len, len(vocab)+1, num_hidden=num_hidden, num_embed=num_embed, num_label=len(vocab)+1, dropout=0.2)


batch_size=32

init_c=[('l%d_init_c'%l, (batch_size, num_hidden)) for l in xrange(num_lstm_layer)]
init_h=[('l%d_init_h'%l, (batch_size, num_hidden)) for l in xrange(num_lstm_layer)]
init_states=init_c+init_h
data_train=bucket_io.BucketSentenceIter('./mldata/obama.txt', vocab, [seq_len],batch_size, init_states, seperate_char='\n', text2id=text2id, read_content=read_content)

num_epoch=1
learning_rate=0.01

model=mx.model.FeedForward(ctx=mx.gpu(0),symbol=symbol, num_epoch=5, learning_rate=learning_rate, initializer=mx.init.Xavier(factor_type='in', magnitude=2.34))
#model=mx.mod.Module(ctx=mx.gpu(0),symbol=symbol, initializer=mx.init.Xavier(factor_type='in', magnitude=2.34))

model.fit(X=data_train, eval_metric=mx.metric.np(Perplexity),batch_end_callback=mx.callback.Speedometer(batch_size,20))


