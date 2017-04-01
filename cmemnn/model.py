import mxnet as mx
import numpy as np
import random
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

def get_cmemnn(batch_size, num_embed, num_hidden, num_layer, num_user, num_item, nupass, nipass, npass, dropout=0.):
    '''
    last_states=[]
    param_cells=[]
    for i in xrange(num_layer):
        param_cells.append(GRUParam(gates_i2h_weight=mx.sym.Variable('l%d_i2h_gates_weight'%i), gates_i2h_bias=mx.sym.Variable('l%d_i2h_gates_bias'%i),gates_h2h_weight=mx.sym.Variable('l%d_h2h_gates_weight'%i),gates_h2h_bias=mx.sym.Variable('l%d_h2h_gates_bias'%i),trans_i2h_weight=mx.sym.Variable('l%d_i2h_trans_weight'%i), trans_i2h_bias=mx.sym.Variable('l%d_i2h_bias'%i), trans_h2h_weight=mx.sym.Variable('l%d_h2h_trans_weight'%i), trans_h2h_bias=mx.sym.Variable('l%d_h2h_bias'%i)))
        state=GRUState(h=mx.sym.Variable('l%d_init_h'%i))
        last_states.append(state)
    '''
    user=mx.sym.Variable('user')
    item=mx.sym.Variable('item')
    rating=mx.sym.Variable('rating')
    grp_u=mx.sym.Variable('grp_u')
    grp_i=mx.sym.Variable('grp_i')

    weight_u=mx.sym.Variable('u_weight')
    weight_i=mx.sym.Variable('i_weight')

    #weight_uu=mx.sym.Variable('aff_u', shape=(num_embed, num_embed), init=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=2))
    #weight_ii=mx.sym.Variable('aff_i', shape=(num_embed, num_embed), init=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=2))

    weight_z=mx.sym.Variable('z_weight')
    bias_z=mx.sym.Variable('z_bias')
    m_u=mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_embed, weight=weight_u)
    m_i=mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_embed, weight=weight_i)

    grp_u= mx.sym.Embedding(data=grp_u, input_dim=num_user, output_dim=num_embed, weight=weight_u)
    grp_i= mx.sym.Embedding(data=grp_i, input_dim=num_item, output_dim=num_embed, weight=weight_i)

    #Col user attention
    att_u=mx.sym.broadcast_mul(mx.sym.expand_dims(m_i, axis=1), grp_u)
    att_u=mx.sym.sum_axis(att_u, axis=2)
    att_u=mx.sym.SoftmaxActivation(data=att_u)
    
    att_u=mx.sym.expand_dims(att_u, axis=2)
    mem_u=mx.sym.broadcast_mul(grp_u, att_u)
    col_u=mx.sym.sum_axis(data=mem_u, axis=1)
    #alpha_u=mx.sym.Variable('alpha_u_weight')
    #m_u=alpha_u*col_u+(1-alpha_u)*m_u
    mu=mx.sym.Concat(*[col_u, m_u], dim=1)

    #Col item attention
    att_i=mx.sym.broadcast_mul(mx.sym.expand_dims(m_u, axis=1), grp_i)
    att_i=mx.sym.sum_axis(att_i, axis=2)
    att_i=mx.sym.SoftmaxActivation(data=att_i)
    
    att_i=mx.sym.expand_dims(att_i, axis=2)
    mem_i=mx.sym.broadcast_mul(grp_i, att_i)
    col_i=mx.sym.sum_axis(data=mem_i, axis=1)
    #alpha_i=mx.sym.Variable('alpha_i_weight')
    #m_i=alpha_i*col_i+(1-alpha_i)*m_i
    mi=mx.sym.Concat(*[col_i, m_i], dim=1)

     
    #pred=col_i*col_u
    #pred=m_u*m_i
    pred=mu*mi
    pred=mx.sym.sum_axis(data=pred, axis=1)
    pred=mx.sym.Flatten(data=pred)
    pred=mx.sym.LinearRegressionOutput(data=pred, label=rating)
    return pred


        







