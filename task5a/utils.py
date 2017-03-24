import json
import collections

def read_content(path):
    with open(path) as f:
        data=json.load(f)
    return data

def build_vocab(data):
    vocab={}
    idx=1
    

def text2id(sentence, vocab):
    words=list(sentence)
    return [vocab[w] for w in words if len(w)]
