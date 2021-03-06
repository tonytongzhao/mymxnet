import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple('LSTMState', ['c','h'])
LSTMParam = namedtuple('LSTMParam',['i2h_weight', 'i2h_bias',
    'h2h_weight','h2h_bias'])
LSTMModel = namedtuple('LSTMModel', ['rnn_exec', 'symbol','init_states','last_states','seq_data','seq_labels','seq_outputs','param_blocks'])


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    #LSTM Cell symbol
    if dropout>0.:
        indata=mx.sym.Dropout(data=indata, p=dropout)
    #Combine Wi, Wf, Wc, Wo into one i2h
    #Combine Ui, Uf, Uc, Uo into one h2h
    i2h=mx.sym.FullyConnected(data=indata, weight=param.i2h_weight, bias=param.i2h_bias, num_hidden=num_hidden*4, name='t%d_l%d_i2h'%(seqidx, layeridx) )
    h2h=mx.sym.FullyConnected(data=prev_state.h, weight=param.h2h_weight, bias=param.h2h_bias,num_hidden=num_hidden*4, name='t%d_l%d_h2h'%(seqidx, layeridx))

    gates=i2h+h2h

    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4, name='t%d_l%d_slice'%(seqidx, layeridx))

    in_gate = mx.sym.Activation(slice_gates[0], act_type='sigmoid')
    forget_gate=mx.sym.Activation(slice_gates[1], act_type='sigmoid')
    out_gate=mx.sym.Activation(slice_gates[2], act_type='sigmoid')

    in_transform = mx.sym.Activation(slice_gates[3], act_type='tanh')

    next_c=(forget_gate* prev_state.c)+(in_gate*in_transform)
    next_h=out_gate*mx.sym.Activation(next_c, act_type='tanh')

    return LSTMState(c=next_c, h=next_h)

def lstm_unroll(num_lstm_layer, seq_len, input_size, num_hidden, num_embed, num_label, dropout=0.):
    embed_weight=mx.sym.Variable('embed_weight')
    cls_weight=mx.sym.Variable('cls_weight')
    cls_bias=mx.sym.Variable('cls_bias')

    param_cells=[]
    last_states=[]
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight'%i), 
                                     i2h_bias = mx.sym.Variable('l%d_i2h_bias'%i),
                                     h2h_weight=mx.sym.Variable('l%d_h2h_weight'%i), 
                                     h2h_bias = mx.sym.Variable('l%d_h2h_bias'%i)
                                     ))
        state=LSTMState(c=mx.sym.Variable('l%d_init_c'%i), h=mx.sym.Variable('l%d_init_h'%i))

        last_states.append(state)

    assert(len(last_states)==num_lstm_layer)

    data=mx.sym.Variable('data')
    hds=mx.sym.Embedding(data=data, weight=embed_weight, input_dim=input_size, output_dim=num_embed, name='w2v_embed')
    w2v=mx.sym.SliceChannel(data=hds,num_outputs=seq_len, squeeze_axis=1, name='w2v_slice')
    loss_all=[]
    for seqidx in xrange(seq_len):
        hidden=w2v[seqidx]
        #Deep LSTM
        for i in xrange(num_lstm_layer):
            if i==0:
                dp_ratio=0.
            else:
                dp_ratio=dropout
            next_state=lstm(num_hidden, indata=hidden, prev_state=last_states[i],param=param_cells[i],seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden=next_state.h
            last_states[i]=next_state
        if dropout:
            hidden=mx.sym.Dropout(data=hidden, p=dropout, name='dropout%d'%(seqidx))
            hidden=mx.sym.BatchNorm(data=hidden, fix_gamma=True, name='bn')
	    loss_all.append(hidden)
    fc=mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias, num_hidden=num_label, name='cls_lstm_fc')
    loss=mx.sym.LinearRegressionOutput(data=fc, label=mx.sym.Variable('label'), name='lstm_sm')    
    return loss 
	#return mx.sym.Group(loss_all)






def lstm_inference_symbol(num_lstm_layer, input_size, num_hidden, num_embed, num_label, dropout=0.):
    
    seqidx=0
    
    embed_weight=mx.sym.Variable('embed_weight')
    cls_weight=mx.sym.Variable('cls_weight')
    cls_bias=mx.sym.Variable('cls_bias')

    param_cells=[]
    last_states=[]
    
    # Param init statement
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight'%i), 
                                     i2h_bias = mx.sym.Variable('l%d_i2h_bias'%i),
                                     h2h_weight=mx.sym.Variable('l%d_h2h_weight'%i), 
                                     h2h_bias = mx.sym.Variable('l%d_h2h_bias'%i)
                                     ))
        state=LSTMState(c=mx.sym.Variable('l%d_init_c'%i), h=mx.sym.Variable('l%d_init_h'%i))

        last_states.append(state)

    assert(len(last_states)==num_lstm_layer)


    data=mx.sym.Variable('data/%d' % seqidx)


    hidden=mx.sym.Embedding(data=data, weight=embed_weight, input_dim=input_size, output_dim=num_embed,name='t%d_embed'%(seqidx))
        
    #Deep LSTM
    for i in xrange(num_lstm_layer):
        if i==0:
            dp_ratio=0.
        else:
            dp_ratio=dropout
        next_state=lstm(num_hidden, indata=hidden, prev_state=last_states[i],param=param_cells[i],seqidx=seqidx, layeridx=i, dropout=dp_ratio)
        hidden=next_state.h
        last_states[i]=next_state

    if dropout:
        hidden=mx.sym.Dropout(data=hidden, p= dropout, name='dropout')
    fc=mx.sym.FullyConnected(data=hidden, weight=cls_weight, bias=cls_bias, num_hidden=num_label, name='lstm_cls_fc')
    sm = mx.sym.Custom(data=fc, label=mx.sym.Variable('label%d'%seqidx), name='t%d_sm'%seqidx, op_type='softmax')
    out=[sm]
    for state in last_states:
        out.append(state.c)
        out.append(state.h)
    return mx.sym.Group(out)





if __name__=='__main__':
    print 'To be tested'





    
