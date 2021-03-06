import mxnet as mx
import random
import bisect
import numpy as np
class BucketFlexIter(mx.io.DataIter):
    def __init__(self, data, label, batch_size, buckets=None, invalid_label=0, data_name='data', label_name='label', dtype='float32'):
        super(BucketFlexIter, self).__init__()
        if not buckets:
            buckets=[i for i, j in enumerate(np.bincount([len(s) for s in data])) if j>=batch_size]
        buckets.sort()

        ndiscard=0
        self.data=[[] for _ in buckets]
		self.label=[[] for _ in buckets]
		for i, sent in enumerate(data):
            buck=bisect.bisect_left(buckets, len(sent))
            if buck==len(buckets):
                ndiscard+=1
                continue
            buff=np.full((buckets[buck],), invalid_label, dtype=dtype)
            buff[:len(sent)]=sent
            self.data[buck].append(buff)
			self.label[buck].append(label[i])
        self.data=[np.asarray(i, dtype=dtype) for i in self.data]
        
        self.batch_size=batch_size
        self.buckets=buckets
        self.data_name=data_name
        self.label_name=label_name
        self.dtype=dtype
        self.invalid_label=invalid_label
        self.nddata=[]
        self.ndlabel=[]
        self.default_bucket_key=max(buckets)
		self.label_size=len(label[0]) 
        self.major_axis=0

        if self.major_axis==0:
            self.provide_data=[(data_name, (batch_size, self.default_bucket_key))]
			self.provide_label=[(label_name,(batch_size, self.label_size)]
        self.idx=[]
        for i, buck in enumerate(self.data):
            self.idx.extend([(i,j) for j in xrange(0, len(buck)-batch_size+1, batch_size)])
        self.curr_idx=0
        self.reset()

    def reset(self):
        self.curr_idx=0
        random.shuffle(self.idx) 
        for buck in self.data:
            np.random.shuffle(buck)
        self.nddata=[]
        self.ndlabel=[]
        for i, buck in enumerate(self.data):
            self.nddata.append(np.array(buck)) 
            self.ndlabel.append(np.array(self.label[i]))

    def next(self):
        if self.curr_idx==len(self.idx):
            raise StopIteration
        i,j=self.idx[self.curr_idx]
        self.curr_idx+=1
        
        data=self.nddata[i][j:j+self.batch_size]
        label=self.ndlabel[i][j:j+self.batch_size]
        return mx.io.DataBatch([mx.nd.array(data)], [mx.nd.array(label)], pad=0, bucket_key=self.buckets[i], provide_data=[(self.data_name, data.shape)], provide_label=[(self.label_name, label.shape)])

