import ijson
import sys, codecs
dfile=sys.argv[1]

#f=open(dfile,'r')

#parser=ijson.parse(f)

class FHoseFile(object):
    def __init__(self, filename, *parms, **kw):
        self.filename = filename

    def iter(self):
        with codecs.open(self.filename, 'r', encoding='utf-8', errors="ignore") as rawjson:
            objs = ijson.items(rawjson, "articles.item")
            for o in objs:
                yield o
objs=FHoseFile(dfile)
i=0
for w in objs.iter():
    i+=1
print i
'''
f=open(dfile,'r')
objs=ijson.items(f, 'articles.item')
print dir(objs)
i=0
while i<1:
    i+=1
    print objs.next()
'''
