import json, ijson, codecs
import collections, requests
import sys, io, bisect
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import RegexpTokenizer
def read_content(path):
    json_data=open(path, 'r').read().decode('utf8', 'ignore').strip()
    data=json.loads(json_data)
    print 'Json loading succeed'
    return data

def read_content_stream(path):
    json_data=codecs.open(path, 'r', encoding='utf8', errors='ignore')
    objs=ijson.items(json_data,'articles.item')
    return objs
def load_data_statics(data, vocab=None, label_dict=None, label_rev_dict=None, tr=True):
    if vocab==None:
    	vocab={}
        label_dict={}
        label_rev_dict={}
    idx=1
    nins=0
    newwords=0
    stop=set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    for obj in data:
        nins+=1
        for k in obj:
            if 'meshMajor' in k:
                for l in obj[k]:
                    if l not in label_dict:
			if not tr:
			    continue
                        label_dict[l]=len(label_dict)
                        label_rev_dict[label_dict[l]]=l
            elif 'abstractText' in k:
                txt=tokenizer.tokenize(obj[k])
                for w in txt:
                    if w in stop:
                        continue
                    if len(w) and w not in vocab:
			if not tr:
                            newwords+=1
			    continue
                        vocab[w]=idx
                        idx+=1
            elif 'title' in k:
                txt=tokenizer.tokenize(obj[k])
                for w in txt:
                    if w in stop:
                        continue
                    if len(w) and w not in vocab:
			if not tr:
                            newwords+=1
			    continue
			vocab[w]=idx
                        idx+=1
    print 'new words', newwords
    print 'cur_words', len(vocab)
    return nins, vocab, label_dict, label_rev_dict

def load_data(data, vocab=None, label_dict=None, label_rev_dict=None, tr=True):
    if vocab==None:
    	vocab={}
        label_dict={}
        label_rev_dict={}
    idx=1
    features=[]
    labels=[]
    pmids=[]
    newwords=0
    stop=set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    if tr:
        t0='articles'
    else:
        t0='documents'
    for d in data[data.keys()[0]]:
        ins_label=[]
        ins_feature=[]
        for k in d:
            if 'meshMajor' in k:
                for l in d[k]:
                    if l not in label_dict:
			if not tr:
			    continue
                        label_dict[l]=len(label_dict)
                    label_rev_dict[label_dict[l]]=l
                    ins_label.append(label_dict[l])
            elif 'abstractText' in k:
                txt=tokenizer.tokenize(d[k])
                for w in txt:
                    if w in stop:
                        continue
                    if len(w) and w not in vocab:
			if not tr:
                            newwords+=1
			    continue
                        vocab[w]=idx
                        idx+=1
                    ins_feature.append(vocab[w])
            elif 'title' in k:
                txt=tokenizer.tokenize(d[k])
                for w in txt:
                    if w in stop:
                        continue
                    if len(w) and w not in vocab:
			if not tr:
                            newwords+=1
			    continue
			vocab[w]=idx
                        idx+=1
                    ins_feature.append(vocab[w])
        features.append(ins_feature)
        labels.append(ins_label)
        pmids.append(d['pmid'])
    print 'new words', newwords
    print 'cur_words', len(vocab)
    return np.array(features), np.array(labels), np.array(pmids), vocab, label_dict, label_rev_dict

def download_test_data(url):
	r=requests.get('http://participants-area.bioasq.org/tests/%s/'%(url), auth=('SanMatrix', 'tsinghua02'))
	#Now the variable r contains the data (can been seen with r.text)
	data=json.loads(r.text)
	return data

def load_test_data(data, vocab):
    idx=1
    feature=[]
    pmid=[]
    stop=set(stopwords.words('english'))
    tokenizer=RegexpTokenizer(r'\w+')
    for d in data['documents']:
        for k in d:
            ins_feature=[]
            if 'pmid' in k:
                pmid.append(d[k])
            elif ('abstract' in k) or ('title' in k):
                txt=tokenizer.tokenize(d[k])
                for w in txt:
                    if w in stop or w not in vocab or not len(w):
                        continue
                    ins_feature.append(vocab[w])
        features.append(ins_feature)
    return np.array(features), np.array(pmid)
	

def mesh_mapping(path):
    mesh_map={}
    mesh_rev_map={}
    with open(path, 'r') as f:
        for line in f:
            ls=line.strip().split('=')
            if ls[1].startswith('D'):
                mesh_map[ls[0]]=ls[1]
                mesh_rev_map[ls[1]]=ls[0]
    return mesh_map, mesh_rev_map

def text2id(sentence, vocab):
    words=list(sentence)
    return [vocab[w] for w in words if len(w)]

def chunkl(n, p):
	m=n/p
	return [m for _ in xrange(p-1)]+[n-(m*(p-1))]

def accuracy(label, pred):
    return len(label==pred)/(0.0+len(pred))

if __name__=='__main__':
	print chunkl(12,10), sum(chunkl(12,10))
