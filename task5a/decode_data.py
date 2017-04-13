import json, ijson, codecs
import collections, requests
import sys, io, bisect
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import RegexpTokenizer
def read_content(path):
    json_data=open(path, 'r').read().decode('utf8', 'ignore').strip()
    data=json.loads(json_data)
    with open(path.split('.json')[0]+'.decode.json', 'w') as of:
        json.dump(data, of)
    print 'Json loading succeed'
    return data

def read_iter(path):
    dw=open(path.split('.json')[0]+'.decode.json', 'w')
    with open(path,'r') as f:
        for line in f:
            line=line.decode('utf8', 'ignore')
            dw.write('%s'%line.encode('ascii', 'replace'))
    dw.close()

df=sys.argv[1]
#read_content(df)
read_iter(df)

