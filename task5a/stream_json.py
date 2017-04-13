import ijson
import sys

dfile=sys.argv[1]

f=open(dfile,'r')
parser=ijson.parse(f)
objs=ijson.items(f,'articles.item')
i=0
while i<1:
    print objs.next()
    i+=1
i=0
while i<1:
    print objs.next()
    i+=1
