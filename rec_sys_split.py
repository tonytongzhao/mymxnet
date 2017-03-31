import sys
import os
import collections
import random
dfile=sys.argv[1]
path='/'.join(dfile.split('/')[:-1])
data=collections.defaultdict(dict)
nins=0
with open(dfile, 'r') as f:
    for line in f:
        tks=line.split('::')
        user,item, rate=tks[0], tks[1], tks[2]
        data[user][item]=rate
        nins+=1
print nins
tr=collections.defaultdict(dict)
te=collections.defaultdict(dict)
ntr=0
nte=0
split=float(sys.argv[2])
for u in data:
    if len(data[u])*split>1.0:
        tr_item=random.sample(data[u].keys(), int(len(data[u])*split))
        te_item=list(set(data[u])-set(tr_item))
    else:
        tr_item=data[u].keys()
        te_item=[]
    tr[u]={i:data[u][i] for i in tr_item}
    te[u]={i:data[u][i] for i in te_item}
    ntr+=len(tr[u])
    nte+=len(te[u])
print ntr
print nte
dftr=path+'/'+dfile.split('/')[-1]+'.tr'
dfte=path+'/'+dfile.split('/')[-1]+'.te'
with open(dftr, 'w') as tfw:
    for u in tr:
        for i in tr[u]:
            tfw.write('%s\t%s\t%s\n' %(u, i, tr[u][i]))
tfw.close()
with open(dfte, 'w') as tfw:
    for u in te:
        for i in te[u]:
            tfw.write('%s\t%s\t%s\n' %(u, i, te[u][i]))
tfw.close()
