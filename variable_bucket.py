import mxnet as mx
import numpy as np
class BucketFlexIter(mx.io.DataIter):
    def __init__(self, data, batch_size, buckets=None, invalid_label=1, data_name='data', label_name='label', dtype='float32'):
        super(BucketFlexIter, self).__init__()
        if not buckets:
            buckets=[i for i, j in enumerate(np.bincount([len(s) for s in data])) if j>=batch_size]
        buckets.sort()

        ndiscard=0
        self.data=[[] for _ in buckets]


