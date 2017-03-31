import mxnet as mx
import numpy as np
from collections import namedtuple

GRUState=namedtuple('GRUState', ['h'])
GRUParam=namedtuple('GRUParam', ['gates_i2h_weight','gates_i2h_bias', 'gates_h2h_weight', 'gates_h2h_bias','trans_i2h_weight','trans_i2h_bias', 'trans_h2h_weight', 'trans_h2h_bias'])


def myGRU(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    if dropout>0.:
        indata=mx.sym.Dropout(data=indata, p=dropout)

    i2h=mx.sym.FullyConnected(data=indata, weight=param.gates_i2h_weight, bias=param.gates_i2h_bias, num_hidden=num_hidden*2, name='t%d_l%d_gates_i2h'%(seqidx, layeridx))
    h2h=mx.sym.FullyConnected(data=prev_state.h, weight=param.gates_h2h_weight, bias=param.gates_h2h_bias, num_hidden=num_hidden*2, name='t%d_l%d_gates_h2h'%(seqidx, layeridx))

    gates=i2h+h2h

    slice_gates=mx.sym.SliceChannel(gates, num_outputs=2, name='t%d_l%d_slice'%(seqidx, layeridx))

    update_gate=mx.sym.Activation(data=slice_gates[0], act_type='sigmoid')
    reset_gate=mx.sym.Activation(data=slice_gates[1], act_type='sigmoid')

    htrans_i2h=mx.sym.FullyConnected(data=indata, weight=param.trans_i2h_weight, bias=param.trans_i2h_bias, num_hidden=num_hidden, name='t%d_l%d_trans_i2h'%(seqidx, layeridx))


    h_after_reset=prev_state.h*reset_gate
    htrans_h2h=mx.sym.FullyConnected(data=h_after_reset, weight=param.trans_h2h_weight, bias=param.trans_h2h_bias, num_hidden=num_hidden, name='t%d_l%d_trans_h2h'%(seqidx, layeridx))

    htrans=htrans_i2h+htrans_h2h

    htrans_act=mx.sym.Activation(data=htrans, act_type='tanh')

    next_h=prev_state.h+update_gate*(htrans_act-prev_state.h)
    return GRUState(h=next_h)

def my_GRU_unroll(num_gru_layer, seq_len, input_size, num_hidden, num_embed, num_label, dropout=0.):
    seqidx=0
    embed_weight=mx.sym.Variable('embed_weight')
    cls_weight=mx.sym.Variable('cls_weight')
    cls_bias=mx.sym.Variable('cls_bias')
    param_cells=[]
    last_states=[]
    
    for i in xrange(num_gru_layer):
        param_cells.append(GRUParam(gates_i2h_weight=mx.sym.Variable('l%d_i2h_gates_weight'%i), gates_i2h_bias=mx.sym.Variable('l%d_i2h_gates_bias'%i),gates_h2h_weight=mx.sym.Variable('l%d_h2h_gates_weight'%i),gates_h2h_bias=mx.sym.Variable('l%d_h2h_gates_bias'%i),trans_i2h_weight=mx.sym.Variable('l%d_i2h_trans_weight'%i), trans_i2h_bias=mx.sym.Variable('l%d_i2h_bias'%i), trans_h2h_weight=mx.sym.Variable('l%d_h2h_trans_weight'%i), trans_h2h_bias=mx.sym.Variable('l%d_h2h_bias'%i)))
        state=GRUState(h=mx.sym.Variable('l%d_init_h'%i))
        last_states.append(state)

    data=mx.sym.Variable('data')
    data=mx.sym.BlockGrad(data)
    label=mx.sym.Variable('label')
    embed=mx.sym.Embedding(data=data, input_dim=input_size, output_dim=num_embed, weight=embed_weight)
    wordvec=mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
    
    hidden_all=[]
    for seqidx in xrange(seq_len):
        hidden=wordvec[seqidx]
        for i in xrange(num_gru_layer):
            if i==0:
                drop_r=0.
            else:
                drop_r=dropout
            next_state=myGRU(num_hidden, indata=hidden, prev_state=last_states[i],param=param_cells[i],seqidx=seqidx, layeridx=i, dropout=drop_r)
            hidden=next_state.h
            last_states[i]=next_state
        if drop_r:
            hidden=mx.sym.Dropout(data=hidden, p=drop_r)
        hidden=mx.sym.BatchNorm(data=hidden, name='bn', fix_gamma=True)
        hidden_all.append(hidden)


    #If seq2seq learning
	'''
	hidden_concat=mx.sym.Concat(*hidden_all, dim=0)
    pred=mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label)
    label=mx.sym.Transpose(data=label)
    label=mx.sym.Reshape(data=label, target_shape=(0,))
    '''
    #If one final output
    fc=mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias, num_hidden=num_label)
    loss=mx.sym.LinearRegressionOutput(data=fc, label=mx.sym.Variable('label'))    
	
    return loss



if __name__=='__main__':
    print 'To be tested'
