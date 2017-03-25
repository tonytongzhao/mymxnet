import json
import collections
import sys, io
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import RegexpTokenizer
def read_content(path):
    json_data=open(path, 'r').read().decode('utf8', 'ignore').strip()
    data=json.loads(json_data)
    print 'Json loading succeed'
    return data

def load_data(data):
    vocab={}
    label_dict={}
    idx=1
    features=[]
    labels=[]
    stop=set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    for d in data['articles']:
        ins_label=[]
        ins_feature=[]
        for k in d:
            if 'meshMajor' in k:
                for l in d[k]:
                    if l not in label_dict:
                        label_dict[l]=len(label_dict)
                    ins_label.append(label_dict[l])
            elif 'abstractText' in k:
                txt=tokenizer.tokenize(d[k])
                for w in txt:
                    if w in stop:
                        continue
                    if len(w) and w not in vocab:
                        vocab[w]=idx
                        idx+=1
                    ins_feature.append(vocab[w])
        features.append(ins_feature)
        labels.append(ins_label)
    return np.array(features), np.array(labels), vocab, label_dict

def text2id(sentence, vocab):
    words=list(sentence)
    return [vocab[w] for w in words if len(w)]

def accuracy(label, pred):
    return len(label==pred)/(0.0+len(pred))
