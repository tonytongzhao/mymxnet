import mxnet as mx
import os, urllib, zipfile
import mylstm
import bucket_io
if not os.path.exists('char_lstm.zip'):
	urllib.urlretrieve('http://data.mxnet.io/data/char_lstm.zip', 'char_lstm.zip')
with zipfile.ZipFile('char_lstm.zip', 'r') as f:
	f.extractall('./')

with open('obama.txt','r') as f:
	print f.read()[0:1000]

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

vocab=build_vocab('./obama.txt')

seq_len=129
num_embed=256
num_lstm_layer=3
num_hidden=512


symbol=mylstm.lstm_unroll(num_lstm_layer, seq_len, len(vocab)+1, num_hidden=num_hidden, num_embed=num_embed, num_label=len(vocab)+1, dropout=0.2)



batch_size=32

init_c=[('l%d_init_c'%l, (batch_size, num_hidden)) for l in ]
