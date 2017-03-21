import mxnet as mx
import os, urllib, zipfile
import mylstm
import numpy as np
import bucket_io
from my_lstm_inference import LSTMInferenceModel
import random
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

def MakeRevertVocab(vocab):
	dic={}
	for k, v in vocab.items():
		dic[v]=k
	return dic

def MakeInput(char, vocab, arr):
	idx=vocab[char]
	tmp=np.zeros((1,))
	tmp[0]=idx
	arr[:]=tmp


def _cdf(weights):
	total=sum(weights)
	result=[]
	cusum=0
	for w in weights:
		cusum+=w
		result.append(cusum/total)
	return result

def _choice(population, weights):
	cdf_vals=_cdf(weights)
	x=random.random()
	idx=bisec.bisec(cdf_vals,x)
	return population[idx]


def MakeOutput(prob, vocab, sample=False, temperature=1.):
	if not sample:
		idx=np.argmax(prob, axis=1)[0]
	else:
		idx=np.random.choice(range(len(prob)))

	try:
		char=vocab[idx]
	except:
		char=''
	return char



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

num_epoch=100
learning_rate=0.01
import logging
model=mx.model.FeedForward(ctx=mx.gpu(0),symbol=symbol, num_epoch=num_epoch, learning_rate=learning_rate, momentum=0,wd=0.0001, initializer=mx.init.Xavier(factor_type='in',magnitude=2.34))
#model=mx.mod.Module(ctx=mx.gpu(0),symbol=symbol, initializer=mx.init.Xavier(factor_type='in', magnitude=2.34))
logging.basicConfig(level=logging.DEBUG)
model.fit(X=data_train, eval_metric=mx.metric.np(Perplexity),batch_end_callback=mx.callback.Speedometer(batch_size,50),epoch_end_callback=mx.callback.do_checkpoint('obama'))

_,arg_params,__=mx.model.load_checkpoint('obama',100)

model=LSTMInferenceModel(num_lstm_layer, len(vocab)+1, num_hidden=num_hidden, num_embed=num_embed,num_label=len(vocab)+1, arg_params=arg_params, ctx=mx.gpu(), dropout=0.2)


seq_length=600
input_ndarray=mx.nd.zeros((1,))
revert_vocab=MakeRevertVocab(vocab)
output='The United States'
print output
random_sample=False
new_sentence=True

ignore_length=len(output)

for i in xrange(seq_length):
	if i<=ignore_length-1:
		MakeInput(output[i], vocab, input_ndarray)
	else:
		MakeInput(output[-1], vocab, input_ndarray)

	prob=model.forward(input_ndarray,new_sentence)
	new_sentence=False
	next_char=MakeOutput(prob,revert_vocab,random_sample)
	if next_char=='':
		new_sentence=True
	if i>=ignore_length-1:
		output+=next_char
print output

