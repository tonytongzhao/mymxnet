import mxnet as mx
from collections import namedtuple
import numpy as np
import sys

LSTMState=namedtuple('LSTMState', ['c', 'h'])
LSTMParam=namedtuple('LSTMParam', ['i2h_weight', 'i2h_bias', 'h2h_weight', 'h2h_bias'])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    if dropout:
        indata=mx.sym.Dropout(data=indata, p=dropout)
    i2h=mx.sym.FullyConnected(data=indata, num_hidden=num_hidden*4, weight=param.i2h_weight, bias=param.i2h_bias, name='t%d_l%d_i2h'%(seqidx, layeridx))
    h2h=mx.sym.FullyConnected(data=prev_state.h, num_hidden=num_hidden*4, weight=param.h2h_weight, bias=param.h2h_bias, name='t%d_l%d_h2h'%(seqidx, layeridx))
    gates=i2h+h2h
    slice_gates=mx.sym.SliceChannel(gates, num_outputs=4)
    in_gate=mx.sym.Activation(data=slice_gates[0], act_type='sigmoid')
    in_transform=mx.sym.Activation(data=slice_gates[1], act_type='tanh')
    forget_gate=mx.sym.Activation(slice_gates[2], act_type='sigmoid')
    out_gate=mx.sym.Activation(slice_gates[3], act_type='sigmoid')

    next_c=(forget_gate*prev_state.c)+(in_gate*in_transform)
    next_h=out_gate*mx.sym.Activation(next_c, act_type='tanh')
    return LSTMState(c=next_c, h=next_h)



def bi_lstm_unroll(indata,seq_len, input_size, num_hidden, num_embed, num_label, dropout=0., layeridx=0):
    embed_weight=mx.sym.Variable('embed_weight')
    
    last_states=[]
    last_states.append(LSTMState(c=mx.sym.Variable('lf%d_init_c'%layeridx), h=mx.sym.Variable('lf%d_init_h')))
    last_states.append(LSTMState(c=mx.sym.Variable('lb%d_init_c'%layeridx), h=mx.sym.Variable('lb%d_init_h'%layeridx)))

    forward_param=LSTMParam(i2h_weight=mx.sym.Variable('lf%d_i2h_weight'%layeridx), i2h_bias=mx.sym.Variable('lf%d_i2h_bias'%layeridx), h2h_weight=mx.sym.Variable('lf%d_h2h_weight'%layeridx), h2h_bias=mx.sym.Variable('lf%d_h2h_bias'%layeridx))
    backward_param=LSTMParam(i2h_weight=mx.sym.Variable('lb%d_i2h_weight'%layeridx), i2h_bias=mx.sym.Variable('lb%d_i2h_bias'%layeridx), h2h_weight=mx.sym.Variable('lb%d_h2h_weight'%layeridx), h2h_bias=mx.sym.Variable('lb%d_h2h_bias'%layeridx))

    forward_hidden=[]
    for seqidx in xrange(seq_len):
        hidden=indata[seqidx]
        next_state=lstm(num_hidden, indata=hidden, prev_state=last_states[0], param=forward_param)
        hidden=next_state.h
        last_states[0]=next_state
        forward_hidden.append(hidden)

    backward_hidden=[]
    for seqidx in xrange(seq_len):
        seqidx=seq_len-seqidx-1
        hidden=indata[seqidx]
        next_state=lstm(num_hidden, indata=hidden, prev_state=last_states[1],param=backward_param)
        hidden=next_state.h
        last_states[1]=next_state
        backward_hidden.append(hidden)
    
    hidden_all=[]
    for i in xrange(seq_len):
        hidden_all.append(mx.sym.FullyConnected(data=mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]],dim=1), num_hidden=num_hidden,weight=mx.sym.Variable('l%d_out_weight'%layeridx),bias=mx.sym.Variable('l%d_out_bias'%layeridx)))
    return hidden_all
    


