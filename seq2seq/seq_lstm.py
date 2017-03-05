import mxnet as mx
import numpy as np
from collections import namedtuple
import sys
import math
import time


LSTMState=namedtuple('LSTMState', ['c','h'])
LSTMParam=namedtuple('LSTMParam', ['i2h_weight', 'i2h_bias', 'h2h_weight', 'h2h_bias'])
LSTMModel=namedtuple('LSTMModel', ['rnn_exec', 'symbol', 'init_states', 'last_states', 'seq_data', 'seq_labels', 'seq_outputs', 'param_blocks'])


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    if dropout:
        indata=mx.sym.Dropout(data=indata, p=dropout)

    i2h=mx.sym.FullyConnected(data=indata, weight=param.i2h_weight, bias=param.i2h_bias, num_hidden=num_hidden*4, name='t%d_l%d_i2h'%(seqidx, layeridx))
    h2h=mx.sym.FullyConnected(data=prev_state.h, weight=param.h2h_weight, bias=param.h2h_bias, num_hidden=num_hidden*4, name='t%d_l%d_h2h'%(seqidx, layeridx))
    gates=i2h+h2h

    slice_gates=mx.sym.SliceChannel(gates, num_outputs=4, name='t%d_l%d_slice'%(seqidx, layeridx))

    in_gate=mx.sym.Activation(slice_gates[0], act_type='sigmoid')
    in_transform = mx.sym.Activation(slice_gates[1], act_type='tanh')
    forget_gate=mx.sym.Activation(slice_gates[2], act_type='sigmoid')
    out_gate=mx.sym.Activation(slice_gates[3], act_type='sigmoid')

    next_c=(forget_gate*prev_state.c)+(in_gate*in_transform)

    next_h=out_gate*mx.sym.Activation(next_c, act_type='tanh')

    return LSTMState(c=next_c, h=next_h)



def encode_lstm_unroll(num_layer, seq_len, num_hidden, dropout=0.):
    param_cells=[]
    last_states=[]
    for i in xrange(num_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight'%(i)), i2h_bias=mx.sym.Variable('l%d_i2h_bias'%i), h2h_weight=mx.sym.Variable('l%d_h2h_weight'%i), h2h_bias=mx.sym.Variable('l%d_h2h_bias'%i)))
        state=LSTMState(c=mx.sym.Variable('l%d_init_c'%i), h=mx.sym.Variable('l%d_init_h'%i))
        last_states.append(state)

    data=mx.sym.Variable('data')
    w2v=mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)
    
    hidden_all=[]
    for seqidx in xrange(seq_len):
        hidden=w2v[seqidx]
        for i in xrange(num_layer):
            if i==0:
                dp_ratio=0.
            else:
                dp_ratio=dropout
            next_state=lstm(num_hidden, hidden, prev_state=last_states[i], param=param_cells[i], seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden=next_state.h
            last_states[i]=next_state
        if dropout:
            hidden=mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat=mx.sym.Concat(*hidden_all, dim=0)
    return hidden


def decode_lstm_unroll(num_layer, seq_len, num_hidden, num_label, dropout=0., is_train=True):
    cls_weight=mx.sym.Variable('cls_weight')
    cls_bias=mx.sym.Variable('cls_bias')
    param_cells=[]
    last_states=[]
    for i in xrange(num_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight'%(i)), i2h_bias=mx.sym.Variable('l%d_i2h_bias'%i), h2h_weight=mx.sym.Variable('l%d_h2h_weight'%i), h2h_bias=mx.sym.Variable('l%d_h2h_bias'%i)))
        state=LSTMState( c=mx.sym.Variable('l%d_init_c'%i),h=mx.sym.Variable('l%d_init_h'%i))
        last_states.append(state)

    data=mx.sym.Variable('data')
    label=mx.sym.Variable('softmax_label')
    w2v=mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

    hidden_all=[]
    for seqidx in xrange(seq_len):
        hidden=w2v[seqidx]
        for i in xrange(num_layer):
            if i==0:
                dp_ratio=0.
            else:
                dp_ratio=dropout
            next_state=lstm(num_hidden, hidden, prev_state=last_states[i], param=param_cells[i], seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden=next_state.h
            last_states[i]=next_state
        if dropout:
            hidden=mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat=mx.sym.Concat(*hidden_all, dim=0)
    pred=mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label, weight=cls_weight, bias=cls_bias, name='pred')

    label=mx.sym.transpose(data=label)
    label=mx.sym.Reshape(data=label, target_shape=(0,))
    sm=mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    return sm

    return hidden

def perplexity(label, pred):
    label=label.T.reshape((-1,))
    loss=0.
    for i in xrange(pred.shape[0]):
        loss+=-np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss/label.size)
