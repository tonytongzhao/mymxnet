import mxnet as mx
import os, urllib, zipfile, json
import lstm,gru, ffn
import numpy as np
from variable_bucket import BucketFlexIter 
from bilstm import bi_lstm_unroll
import argparse, bisect
import random, string
from utils import read_content, load_data, text2id, download_test_data, load_test_data, mesh_mapping, chunkl, read_content_stream, load_data_statics
import numpy as np
import logging
def accuracy(label, pred):
    '''
    print 'hi'
    print label[0][:100]
    print np.round(pred[0][:100])
    print sum(label[0][:100])
    print sum(np.round(pred[0][:100]))
    print sum(np.multiply(label[0][:100],np.round(pred[0][:100])))
    '''
    prec=0.0
    batch_size=pred.shape[0]
    for i in xrange(batch_size):
        l=set(np.nonzero(label[i])[0])
        p=set(np.argsort(pred[i])[-50:])
        prec+=len(p & l)
    return prec/np.sum(label)
    #return np.sum(label)

def ins_recall(label, pred):
#    return np.sum(np.multiply(label,np.round(pred))!=0)/np.sum(np.round(pred))
    prec=0.0
    batch_size=pred.shape[0]
    for i in xrange(batch_size):
        l=set(np.nonzero(label[i])[0])
        p=set(np.argsort(pred[i])[-50:])
        prec+=len(p & l)
    return prec/(50.0*batch_size)



def Perplexity(label, pred):
    loss=0.
    for i in range(pred.shape[0]):
        loss+=-np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss/label.size)

def get_data_iter(ins, labels, nlabels, batch_size, init_states, buckets, split):
    num_ins=len(ins)
    tr=random.sample(range(num_ins), int(split*num_ins))
    te=list(set(range(num_ins))-set(tr))
    return BucketFlexIter(ins[tr], labels[tr], nlabels, batch_size, init_states, buckets), BucketFlexIter(ins[te], labels[te], nlabels, batch_size, init_states, buckets) 

def train(args, path, df, val, te, meshmap, nhidden, nembed, batch_size, nepoch, model, nlayer, eta, dropout, split, is_train):
    assert model in ['ffn', 'lstm', 'bilstm', 'gru']
    data=read_content_stream(os.path.join(path, df))
    nins, vocab, label_dict, label_rev_dict = load_data_statics(data)
    mesh_map, mesh_rev_map=mesh_mapping(meshmap)
    contexts=[mx.context.gpu(i) for i in xrange(1)]   
    nwords=len(vocab)
    nlabels=len(label_dict)
    print '#ins', nins
    print '#labels', nlabels
    print '#words', nwords
    npart=30
    pins=chunkl(nins, npart)
    buckets=[50, 100,200, 300, 150, 1000]
    prefix=model+'_'+str(nlayer)+'_'+str(nhidden)+"_"+str(nembed)
    gen_data=read_content_stream(os.path.join(path, df)) 
    logging.basicConfig(level=logging.DEBUG)
    logging.info('start with arguments %s', args)
    if model=='ffn':
        def ffn_gen(seq_len):
            sym=ffn.ffn(nlayer, seq_len, nwords, nhidden, nembed, nlabels, dropout)
	    data_names=['data']
	    label_names=['label']
	    return sym, data_names, label_names
        for pidx in xrange(len(pins)):   
            print 'partition ',pidx
            data={'articles':[]} 
            for _ in xrange(pins[pidx]):
                data['articles'].append(gen_data.next())
            if val==None:
                tr_data, val_data=get_data_iter(ins, labels, nlabels, batch_size,[], buckets, split)
            else:
                ins,labels, pmids, v,ld,lrd=load_data(data, vocab, label_dict, label_rev_dict)
                tr_data=BucketFlexIter(ins, labels, nlabels, batch_size, [], buckets)
                vins,vlabels, vpmids, v,ld,lrd=load_data(read_content(os.path.join(path,val)),vocab, label_dict, label_rev_dict, tr=False)
                val_data=BucketFlexIter(vins, vlabels, nlabels, batch_size, [], buckets)
       	    if len(buckets) == 1:
	        mod = mx.mod.Module(*ffn_gen(buckets[0]), context=contexts)
            else:
	        mod = mx.mod.BucketingModule(ffn_gen, default_bucket_key=tr_data.default_bucket_key, context=contexts) 
            if is_train:
                if pidx:
                    sym, arg_params, aux_params = mx.model.load_checkpoint('./models/%s-%s'%(prefix,pidx-1), nepoch)
                    mod.bind(data_shapes=tr_data.provide_data, label_shapes=tr_data.provide_label, for_training=True)
                    mod.set_params(arg_params=arg_params, aux_params=aux_params)
                    mod.fit(tr_data, eval_data=val_data, num_epoch=nepoch, epoch_end_callback=mx.callback.do_checkpoint('./models/%s-%s'%(prefix,pidx), period=nepoch), eval_metric=['rmse', accuracy, ins_recall])
                else:
                    mod.fit(tr_data, eval_data=val_data, num_epoch=nepoch, epoch_end_callback=mx.callback.do_checkpoint('./models/%s-%s' % (prefix, pidx), period=nepoch), eval_metric=['rmse', accuracy, ins_recall],batch_end_callback=mx.callback.Speedometer(batch_size, 500),initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), optimizer='sgd', optimizer_params={'learning_rate':eta, 'momentum': 0.9, 'wd': 0.00001})
    elif model =='lstm':
        init_c = [('l%d_init_c'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_h = [('l%d_init_h'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_states = init_c + init_h
	state_names=[x[0] for x in init_states]
	def lstm_gen(seq_len):
            sym=lstm.lstm_unroll(nlayer, seq_len, nwords, nhidden, nembed, nlabels, dropout)
	    data_names=['data']+state_names
	    label_names=['label']
	    return sym, data_names, label_names
    
        for pidx in xrange(len(pins)):   
            print 'partition ',pidx
            data={'articles':[]} 
            for _ in xrange(pins[pidx]):
                data['articles'].append(gen_data.next())
            if val==None:
                tr_data, val_data=get_data_iter(ins, labels, nlabels, batch_size,[], buckets, split)
            else:
                ins,labels, pmids, v,ld,lrd=load_data(data, vocab, label_dict, label_rev_dict)
                tr_data=BucketFlexIter(ins, labels, nlabels, batch_size, [], buckets)
                vins,vlabels, vpmids, v,ld,lrd=load_data(read_content(os.path.join(path,val)),vocab, label_dict, label_rev_dict, tr=False)
                val_data=BucketFlexIter(vins, vlabels, nlabels, batch_size, [], buckets)
       	    if len(buckets) == 1:
	        mod = mx.mod.Module(*lstm_gen(buckets[0]), context=contexts)
            else:
	        mod = mx.mod.BucketingModule(lstm_gen, default_bucket_key=tr_data.default_bucket_key, context=contexts) 
            if is_train:
                if pidx:
                    sym, arg_params, aux_params = mx.model.load_checkpoint('./models/%s-%s'%(prefix,pidx-1), nepoch)
                    mod.bind(data_shapes=tr_data.provide_data, label_shapes=tr_data.provide_label, for_training=True)
                    mod.set_params(arg_params=arg_params, aux_params=aux_params)
                    mod.fit(tr_data, eval_data=val_data, num_epoch=nepoch, epoch_end_callback=mx.callback.do_checkpoint('./models/%s-%s'%(prefix,pidx), period=nepoch), eval_metric=['rmse', accuracy, ins_recall])
                else:
                    mod.fit(tr_data, eval_data=val_data, num_epoch=nepoch, epoch_end_callback=mx.callback.do_checkpoint('./models/%s-%s' % (prefix, pidx), period=nepoch), eval_metric=['rmse', accuracy, ins_recall],batch_end_callback=mx.callback.Speedometer(batch_size, 500),initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), optimizer='sgd', optimizer_params={'learning_rate':eta, 'momentum': 0.9, 'wd': 0.00001})
   
    
    elif model=='gru':
        init_h = [('l%d_init_h'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_states = init_h
	state_names=[x[0] for x in init_states]
        def gru_gen(seq_len):
            sym=gru.my_GRU_unroll(nlayer, seq_len, nwords, nhidden, nembed, nlabels, dropout)
	    data_names=['data']+state_names
	    label_names=['label']
	    return sym, data_names, label_names
        for pidx in xrange(len(pins)):   
            print 'partition ',pidx
            data={'articles':[]} 
            for _ in xrange(pins[pidx]):
                data['articles'].append(gen_data.next())
            if val==None:
                tr_data, val_data=get_data_iter(ins, labels, nlabels, batch_size,[], buckets, split)
            else:
                ins,labels, pmids, v,ld,lrd=load_data(data, vocab, label_dict, label_rev_dict)
                tr_data=BucketFlexIter(ins, labels, nlabels, batch_size, [], buckets)
                vins,vlabels, vpmids, v,ld,lrd=load_data(read_content(os.path.join(path,val)),vocab, label_dict, label_rev_dict, tr=False)
                val_data=BucketFlexIter(vins, vlabels, nlabels, batch_size, [], buckets)
       	    if len(buckets) == 1:
	        mod = mx.mod.Module(*lstm_gen(buckets[0]), context=contexts)
            else:
	        mod = mx.mod.BucketingModule(lstm_gen, default_bucket_key=tr_data.default_bucket_key, context=contexts) 
            if is_train:
                if pidx:
                    sym, arg_params, aux_params = mx.model.load_checkpoint('./models/%s-%s'%(prefix,pidx-1), nepoch)
                    mod.bind(data_shapes=tr_data.provide_data, label_shapes=tr_data.provide_label, for_training=True)
                    mod.set_params(arg_params=arg_params, aux_params=aux_params)
                    mod.fit(tr_data, eval_data=val_data, num_epoch=nepoch, epoch_end_callback=mx.callback.do_checkpoint('./models/%s-%s'%(prefix,pidx), period=nepoch), eval_metric=['rmse', accuracy, ins_recall])
                else:
                    mod.fit(tr_data, eval_data=val_data, num_epoch=nepoch, epoch_end_callback=mx.callback.do_checkpoint('./models/%s-%s' % (prefix, pidx), period=nepoch), eval_metric=['rmse', accuracy, ins_recall],batch_end_callback=mx.callback.Speedometer(batch_size, 500),initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), optimizer='sgd', optimizer_params={'learning_rate':eta, 'momentum': 0.9, 'wd': 0.00001})
   

    elif model=='bilstm':
        init_cf = [('lf%d_init_c'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_cb = [('lb%d_init_c'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_hf = [('lf%d_init_h'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_hb = [('lb%d_init_h'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_states = init_cf + init_hf + init_cb + init_hb
	state_names=[x[0] for x in init_states]
	def bilstm_gen(seq_len):
            data=mx.sym.Variable('data')
            embed_weight=mx.sym.Variable('embed_weight')
            concat_weight=mx.sym.Variable('concat_weight')
            hds=mx.sym.Embedding(data=data, weight=embed_weight, input_dim=nwords, output_dim=nembed, name='embed')
            w2v=mx.sym.SliceChannel(data=hds, num_outputs=seq_len,squeeze_axis=1)
            for layidx in xrange(nlayer):
                w2v=bi_lstm_unroll(w2v, concat_weight, seq_len, nwords, nhidden, nembed, nlabels, dropout, layidx)
            w2v=[mx.sym.expand_dims(x, axis=1) for x in w2v]
            hidden=mx.sym.Concat(*w2v, dim=1)
	    hidden=mx.sym.sum_axis(hidden, axis=1)/seq_len
            cls_weight=mx.sym.Variable('cls_weight')
            cls_bias=mx.sym.Variable('cls_bias')
            hidden=mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias, num_hidden=nlabels, name='fc_cls')
            loss=mx.sym.LinearRegressionOutput(data=hidden, label=mx.sym.Variable('label'))
            return loss, ['data']+state_names, ['label']
        for pidx in xrange(len(pins)):   
            print 'partition ',pidx
            data={'articles':[]} 
            for _ in xrange(pins[pidx]):
                data['articles'].append(gen_data.next())
            if val==None:
                tr_data, val_data=get_data_iter(ins, labels, nlabels, batch_size,[], buckets, split)
            else:
                ins,labels, pmids, v,ld,lrd=load_data(data, vocab, label_dict, label_rev_dict)
                tr_data=BucketFlexIter(ins, labels, nlabels, batch_size, [], buckets)
                vins,vlabels, vpmids, v,ld,lrd=load_data(read_content(os.path.join(path,val)),vocab, label_dict, label_rev_dict, tr=False)
                val_data=BucketFlexIter(vins, vlabels, nlabels, batch_size, [], buckets)
       	    if len(buckets) == 1:
	        mod = mx.mod.Module(*lstm_gen(buckets[0]), context=contexts)
            else:
	        mod = mx.mod.BucketingModule(lstm_gen, default_bucket_key=tr_data.default_bucket_key, context=contexts) 
            if is_train:
                if pidx:
                    sym, arg_params, aux_params = mx.model.load_checkpoint('./models/%s-%s'%(prefix,pidx-1), nepoch)
                    mod.bind(data_shapes=tr_data.provide_data, label_shapes=tr_data.provide_label, for_training=True)
                    mod.set_params(arg_params=arg_params, aux_params=aux_params)
                    mod.fit(tr_data, eval_data=val_data, num_epoch=nepoch, epoch_end_callback=mx.callback.do_checkpoint('./models/%s-%s'%(prefix,pidx), period=nepoch), eval_metric=['rmse', accuracy, ins_recall])
                else:
                    mod.fit(tr_data, eval_data=val_data, num_epoch=nepoch, epoch_end_callback=mx.callback.do_checkpoint('./models/%s-%s' % (prefix, pidx), period=nepoch), eval_metric=['rmse', accuracy, ins_recall],batch_end_callback=mx.callback.Speedometer(batch_size, 500),initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), optimizer='sgd', optimizer_params={'learning_rate':eta, 'momentum': 0.9, 'wd': 0.00001})
      




    return vocab, label_dict, label_rev_dict, prefix, buckets, mesh_map, mesh_rev_map

def predict(te, vocab ,label_dict, label_rev_dict, mesh_map, mesh_rev_map, prefix, buckets, model, nhidden, nlayer, dropout, nepoch, batch_size):
    # Prediction for testing data set
    batch_size=1
    tins, tlabels, tpmids, t,tld,tlrd=load_data(read_content(te), vocab, label_dict, label_rev_dict, tr=False)	
    print 'tins', len(tins)
    res={}
    res["documents"]=[]
    param_file = "./models/%s-%s" % (prefix, 30)
    #arg_param,aux_param=load_param(param_file)    
    make_predict(res, tins, len(label_dict), tpmids, model, param_file, buckets, nhidden, nlayer, vocab, dropout, label_rev_dict, mesh_map, mesh_rev_map,nepoch, batch_size)
    return res

def load_param(param_path):
    node_data=mx.nd.load(param_path)
    arg_param={}
    aux_param={}
    for k, v in node_data.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_param[name] = v
        if tp == 'aux':
            aux_param[name] = v
    print(arg_param)
    print(aux_param)
    return arg_param, aux_param

def make_predict(res, test_data, nlabels, test_pmid, model, param_path, buckets, nhidden, nlayer, vocab, dropout, label_rev_dict, mesh_map, mesh_rev_map, nepoch, batch_size):
    dummy_labels=[0 for _ in xrange(len(test_data))]
    if model=='lstm': 
        init_c = [('l%d_init_c'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_h = [('l%d_init_h'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_states = init_c + init_h
        test_iter=BucketFlexIter(test_data, dummy_labels, nlabels, batch_size, init_states, buckets)
        def lstm_gen(seq_len):
            sym=lstm.lstm_unroll(nlayer, seq_len, len(vocab), nhidden, nembed, nlabels, dropout)
	    data_names=['data']+state_names
	    label_names=['label']
	    return sym, data_names, label_names
        module = mx.mod.BucketingModule(lstm_gen, default_bucket_key=test_iter.default_bucket_key)
                                    
    elif model=='ffn': 
        test_iter=BucketFlexIter(test_data, dummy_labels, nlabels, batch_size, [], buckets)
        def ffn_gen(seq_len):
            sym=ffn.ffn(nlayer, seq_len, len(vocab), nhidden, nembed, nlabels, dropout)
	    data_names=['data']
	    label_names=['label']
	    return sym, data_names, label_names
        module = mx.mod.BucketingModule(ffn_gen, default_bucket_key=test_iter.default_bucket_key)
    elif model=='gru': 
        init_h = [('l%d_init_h'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_states = init_h
	state_names=[x[0] for x in init_states]
        test_iter=BucketFlexIter(test_data, dummy_labels, nlabels, batch_size, init_states, buckets)
        def gru_gen(seq_len):
            sym=gru.my_GRU_unroll(nlayer, seq_len, len(vocab), nhidden, nembed, nlabels, dropout)
	    data_names=['data']+state_names
	    label_names=['label']
	    return sym, data_names, label_names
        module = mx.mod.BucketingModule(gru_gen, default_bucket_key=test_iter.default_bucket_key)
    elif model=='bilstm':
        init_cf = [('lf%d_init_c'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_cb = [('lb%d_init_c'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_hf = [('lf%d_init_h'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_hb = [('lb%d_init_h'%l, (batch_size, nhidden)) for l in range(nlayer)]
        init_states = init_cf + init_hf + init_cb + init_hb
	state_names=[x[0] for x in init_states]
        test_iter=BucketFlexIter(test_data, dummy_labels, nlabels, batch_size, init_states, buckets)
	def bilstm_gen(seq_len):
            data=mx.sym.Variable('data')
            embed_weight=mx.sym.Variable('embed_weight')
            concat_weight=mx.sym.Variable('concat_weight')
            hds=mx.sym.Embedding(data=data, weight=embed_weight, input_dim=nwords, output_dim=nembed, name='embed')
            w2v=mx.sym.SliceChannel(data=hds, num_outputs=seq_len,squeeze_axis=1)
            for layidx in xrange(nlayer):
                w2v=bi_lstm_unroll(w2v, concat_weight, seq_len, nwords, nhidden, nembed, nlabels, dropout, layidx)
            w2v=[mx.sym.expand_dims(x, axis=1) for x in w2v]
            hidden=mx.sym.Concat(*w2v, dim=1)
	    hidden=mx.sym.sum_axis(hidden, axis=1)/seq_len
            cls_weight=mx.sym.Variable('cls_weight')
            cls_bias=mx.sym.Variable('cls_bias')
            hidden=mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias, num_hidden=nlabels, name='fc_cls')
            loss=mx.sym.LinearRegressionOutput(data=hidden, label=mx.sym.Variable('label'))
            return loss, ['data']+state_names, ['label']
        if len(buckets) == 1:
	    mod = mx.mod.Module(*bilstm_gen(buckets[0]), context=contexts)
        else:
	    mod = mx.mod.BucketingModule(bilstm_gen, default_bucket_key=tr_data.default_bucket_key, context=contexts)
    module.bind(data_shapes=test_iter.provide_data, label_shapes=None, for_training=False)
    #arg_params, aux_params=load_param(param_path)
    sym, arg_params, aux_params = mx.model.load_checkpoint(param_path, nepoch)
    module.set_params(arg_params=arg_params, aux_params=aux_params)
    unique={}
    total=0
    for preds, i_batch, batch in module.iter_predict(test_iter):
        label = batch.label[0].asnumpy().astype('int32')
        posteriors = preds[0].asnumpy().astype('float32')
        idx=batch.index
        i,j=test_iter.idx[idx]
        batch_ids=test_iter.batch2id[i][j:j+batch_size]
        total+=len(batch_ids)
        pmids=test_pmid[batch_ids]
        for p in pmids:
            if p in unique:
                print 'error'
                break
            unique[p]=1
        ntest=posteriors.shape[0]
        for insidx in xrange(ntest):
            ins_dict={}
            ins_dict["pmid"]=pmids[insidx]
            ins_dict["labels"]=[mesh_map[label_rev_dict[k]] for k in np.argsort(posteriors[insidx,:])[-50:]]
            #print len(ins_dict["labels"])
            res["documents"].append(ins_dict)
    print 'pred', total
    return res

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-path', help='data path', dest='path', required=True)
    parser.add_argument('-file', help='data file', dest='fi', required=True)
    parser.add_argument('-val', help='data val', dest='val', default=None)
    parser.add_argument('-test', help='data test', dest='te', default=None)
    parser.add_argument('-map', help='Mesh Mapping', dest='mapping')
    parser.add_argument('-nhidden', help='num of hidden', dest='num_hidden', default=50)
    parser.add_argument('-nembed', help='num of embedding', dest='num_embed', default=50)
    parser.add_argument('-batch_size', help='batch size', dest='batch_size', default=100)
    parser.add_argument('-nepoch', help='num of epoch', dest='num_epoch', default=200)
    parser.add_argument('-nlayer', help='num of GRU layers', dest='num_layer', default=1)
    parser.add_argument('-eta', help='learning rate', dest='learning_rate', default=0.005)
    parser.add_argument('-dropout', help='dropout', dest='dropout', default=0.2)
    parser.add_argument('-split',dest='split', help='train & validation split ratio', default=0.9)
    parser.add_argument('-model', dest='model', help='model module: ffn, lstm, bilstm, gru', required=True)
    parser.add_argument('-is_train', dest='is_train', default=1)
    args=parser.parse_args()
    path=args.path
    df=args.fi
    te=args.te
    val=args.val
    mesh_map=args.mapping
    nhidden=int(args.num_hidden)
    nembed=int(args.num_embed)
    batch_size=int(args.batch_size)
    nepoch=int(args.num_epoch)
    model=args.model
    nlayer=int(args.num_layer)
    eta=float(args.learning_rate)
    dropout=float(args.dropout)
    split=float(args.split)
    is_train=int(args.is_train)
    vocab, label_dict, label_rev_dict, prefix, buckets, mesh_map, mesh_rev_map = train(args, path, df, val, te, mesh_map, nhidden, nembed, batch_size, nepoch, model, nlayer, eta, dropout, split, is_train)
    print 'Prediction begins...'
    res=predict(te, vocab, label_dict, label_rev_dict, mesh_map, mesh_rev_map, prefix, buckets, model, nhidden, nlayer, dropout, nepoch, batch_size)
    print 'Writing results...'    
    with open('./pred/'+te.split('/')[-1].split('.json')[0]+"_"+prefix+'.json', 'w') as outfile:
        json.dump(res, outfile)









