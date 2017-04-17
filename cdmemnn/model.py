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

def get_cdmemnn(batch_size, num_embed, num_hidden, num_layer, num_user, num_item, nupass, nipass, npass, dropout=0.):
    last_states=[]
    param_ucells=[]
    param_icells=[]
    param_cells=[]
    for i in xrange(num_layer):
        param_ucells.append(GRUParam(gates_i2h_weight=mx.sym.Variable('ul%d_i2h_gates_weight'%i), gates_i2h_bias=mx.sym.Variable('ul%d_i2h_gates_bias'%i),gates_h2h_weight=mx.sym.Variable('ul%d_h2h_gates_weight'%i),gates_h2h_bias=mx.sym.Variable('ul%d_h2h_gates_bias'%i),trans_i2h_weight=mx.sym.Variable('ul%d_i2h_trans_weight'%i), trans_i2h_bias=mx.sym.Variable('ul%d_i2h_bias'%i), trans_h2h_weight=mx.sym.Variable('ul%d_h2h_trans_weight'%i), trans_h2h_bias=mx.sym.Variable('ul%d_h2h_bias'%i)))
        param_icells.append(GRUParam(gates_i2h_weight=mx.sym.Variable('il%d_i2h_gates_weight'%i), gates_i2h_bias=mx.sym.Variable('il%d_i2h_gates_bias'%i),gates_h2h_weight=mx.sym.Variable('il%d_h2h_gates_weight'%i),gates_h2h_bias=mx.sym.Variable('il%d_h2h_gates_bias'%i),trans_i2h_weight=mx.sym.Variable('il%d_i2h_trans_weight'%i), trans_i2h_bias=mx.sym.Variable('il%d_i2h_bias'%i), trans_h2h_weight=mx.sym.Variable('il%d_h2h_trans_weight'%i), trans_h2h_bias=mx.sym.Variable('il%d_h2h_bias'%i)))
        param_cells.append(GRUParam(gates_i2h_weight=mx.sym.Variable('l%d_i2h_gates_weight'%i), gates_i2h_bias=mx.sym.Variable('l%d_i2h_gates_bias'%i),gates_h2h_weight=mx.sym.Variable('l%d_h2h_gates_weight'%i),gates_h2h_bias=mx.sym.Variable('l%d_h2h_gates_bias'%i),trans_i2h_weight=mx.sym.Variable('l%d_i2h_trans_weight'%i), trans_i2h_bias=mx.sym.Variable('l%d_i2h_bias'%i), trans_h2h_weight=mx.sym.Variable('l%d_h2h_trans_weight'%i), trans_h2h_bias=mx.sym.Variable('l%d_h2h_bias'%i)))
        state=GRUState(h=mx.sym.Variable('l%d_init_h'%i))
        last_states.append(state)
    user=mx.sym.Variable('user')
    item=mx.sym.Variable('item')
    rating=mx.sym.Variable('rating')
    grp_u=mx.sym.Variable('grp_u')
    grp_i=mx.sym.Variable('grp_i')
    
    weight_emu=mx.sym.Variable('emu_weight')
    weight_emi=mx.sym.Variable('emi_weight')

    weight_u=mx.sym.Variable('u_weight')
    weight_i=mx.sym.Variable('i_weight')

    #weight_uu=mx.sym.Variable('aff_u', shape=(num_embed, num_embed), init=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=2))
    #weight_ii=mx.sym.Variable('aff_i', shape=(num_embed, num_embed), init=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=2))

    weight_z1=mx.sym.Variable('z1_weight')
    weight_z2=mx.sym.Variable('z2_weight')
    weight_z3=mx.sym.Variable('z3_weight')
    bias_z1=mx.sym.Variable('z1_bias')
    bias_z2=mx.sym.Variable('z2_bias')
    bias_z3=mx.sym.Variable('z3_bias')
    m_u=mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_embed, weight=weight_u)
    m_i=mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_embed, weight=weight_i)

    grp_u= mx.sym.Embedding(data=grp_u, input_dim=num_user, output_dim=num_embed, weight=weight_u)
    grp_i= mx.sym.Embedding(data=grp_i, input_dim=num_item, output_dim=num_embed, weight=weight_i)

    col_u=mx.sym.SliceChannel(data=grp_u, axis=1, num_outputs=nupass, squeeze_axis=1)
    col_i=mx.sym.SliceChannel(data=grp_i, axis=1, num_outputs=nipass, squeeze_axis=1)
    
    mu=[m_u]
    mi=[m_i]
    for _ in xrange(npass):
        for i in xrange(nupass):
            cur_col_u=col_u[i]
            #cur_col_u=mx.sym.BlockGrad(data=cur_col_u)
            q_u=mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_embed, weight=weight_u)
            z=[cur_col_u, m_u, q_u,  mx.sym.abs(q_u-cur_col_u), mx.sym.abs(cur_col_u-m_u)]
            #z=[ mx.sym.abs(q_u-m_u), mx.sym.abs(q_u-cur_col_u), mx.sym.abs(cur_col_u-m_u)]
            z=mx.sym.Concat(*z, dim=1)
            z=mx.sym.FullyConnected(data=z, num_hidden=num_hidden,weight=weight_z1, bias=bias_z1)
            z=mx.sym.Activation(data=z, act_type='relu')
            z=mx.sym.BatchNorm(data=z, fix_gamma=True)
            z=mx.sym.FullyConnected(data=z, num_hidden=nupass, weight=weight_z2, bias=bias_z2)
            z=mx.sym.Activation(data=z, act_type='sigmoid')
            z=mx.sym.FullyConnected(data=z, num_hidden=num_embed, weight=weight_z3, bias=bias_z3)
            gu=mx.sym.SoftmaxActivation(data=z)
            mu_state=GRUState(h=m_u)
            next_state=myGRU(num_embed, indata=cur_col_u, prev_state=mu_state, param=param_ucells[0], seqidx=0, layeridx=0, dropout=dropout)
            m_u=(next_state.h-m_u)*gu+m_u
        e_u=m_u 
        next_s=myGRU(num_embed, indata=e_u, prev_state=GRUState(h=mu[-1]), param=param_cells[0], seqidx=0, layeridx=0, dropout=dropout)
        mu.append(next_s.h)
            
        for i in xrange(nipass):
            cur_col_i=col_i[i]
            #cur_col_i=mx.sym.BlockGrad(data=cur_col_i)
            q_i=mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_embed, weight=weight_i)
            z=[cur_col_i, m_i, q_i, mx.sym.abs(q_i-cur_col_i), mx.sym.abs(cur_col_i-m_i)]
            #z=[mx.sym.abs(q_i-m_i), mx.sym.abs(q_i-cur_col_i), mx.sym.abs(cur_col_i-m_i)]
            z=mx.sym.Concat(*z, dim=1)
            z=mx.sym.FullyConnected(data=z, num_hidden=num_hidden,weight=weight_z1, bias=bias_z1)
            z=mx.sym.Activation(data=z, act_type='relu')
            z=mx.sym.BatchNorm(data=z, fix_gamma=True)
            z=mx.sym.FullyConnected(data=z, num_hidden=nipass, weight=weight_z2, bias=bias_z2)
            z=mx.sym.Activation(data=z, act_type='sigmoid')
            z=mx.sym.FullyConnected(data=z, num_hidden=nipass, weight=weight_z3, bias=bias_z3)
            gi=mx.sym.SoftmaxActivation(data=z)
            mi_state=GRUState(h=m_i)
            next_state=myGRU(num_embed, indata=cur_col_i,prev_state=mi_state, param=param_icells[0], seqidx=0, layeridx=0, dropout=dropout)
            m_i=(next_state.h-m_i)*gu+m_i
        e_i=m_i
        next_s=myGRU(num_embed, indata=e_i, prev_state=GRUState(h=mi[-1]), param=param_cells[0], seqidx=0, layeridx=0, dropout=dropout)
        mi.append(next_s.h)
    m_u=mu[-1]
    m_i=mi[-1]
    m_u=mx.sym.Concat(m_u, mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_embed, weight=weight_emu), dim=1)
    m_i=mx.sym.Concat(m_i, mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_embed, weight=weight_emi), dim=1)
    
    pred=m_u*m_i
    pred=mx.sym.sum_axis(data=pred, axis=1)
    pred=mx.sym.Flatten(data=pred)
    pred=mx.sym.LinearRegressionOutput(data=pred, label=rating)
#    pred=mx.sym.FullyConnected(data=pred, num_hidden=5, name='cls')   
#    pred=mx.sym.SoftmaxOutput(data=pred, label=rating)
    return pred


        


def get_colgru(batch_size, num_embed, num_hidden, num_layer, num_user, num_item, nupass, nipass, npass, dropout=0.):
    last_states=[]
    param_ucells=[]
    param_icells=[]
    for i in xrange(num_layer):
        param_ucells.append(GRUParam(gates_i2h_weight=mx.sym.Variable('ul%d_i2h_gates_weight'%i), gates_i2h_bias=mx.sym.Variable('ul%d_i2h_gates_bias'%i),gates_h2h_weight=mx.sym.Variable('ul%d_h2h_gates_weight'%i),gates_h2h_bias=mx.sym.Variable('ul%d_h2h_gates_bias'%i),trans_i2h_weight=mx.sym.Variable('ul%d_i2h_trans_weight'%i), trans_i2h_bias=mx.sym.Variable('ul%d_i2h_bias'%i), trans_h2h_weight=mx.sym.Variable('ul%d_h2h_trans_weight'%i), trans_h2h_bias=mx.sym.Variable('ul%d_h2h_bias'%i)))
        param_icells.append(GRUParam(gates_i2h_weight=mx.sym.Variable('il%d_i2h_gates_weight'%i), gates_i2h_bias=mx.sym.Variable('il%d_i2h_gates_bias'%i),gates_h2h_weight=mx.sym.Variable('il%d_h2h_gates_weight'%i),gates_h2h_bias=mx.sym.Variable('il%d_h2h_gates_bias'%i),trans_i2h_weight=mx.sym.Variable('il%d_i2h_trans_weight'%i), trans_i2h_bias=mx.sym.Variable('il%d_i2h_bias'%i), trans_h2h_weight=mx.sym.Variable('il%d_h2h_trans_weight'%i), trans_h2h_bias=mx.sym.Variable('il%d_h2h_bias'%i)))
        state=GRUState(h=mx.sym.Variable('l%d_init_h'%i))
        last_states.append(state)
    user=mx.sym.Variable('user')
    item=mx.sym.Variable('item')
    rating=mx.sym.Variable('rating')
    grp_u=mx.sym.Variable('grp_u')
    grp_i=mx.sym.Variable('grp_i')
    
    weight_emu=mx.sym.Variable('emu_weight')
    weight_emi=mx.sym.Variable('emi_weight')

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

    col_u=mx.sym.SliceChannel(data=grp_u, axis=1, num_outputs=nupass, squeeze_axis=1)
    col_i=mx.sym.SliceChannel(data=grp_i, axis=1, num_outputs=nipass, squeeze_axis=1)
    for _ in xrange(npass):
        for i in xrange(nupass):
            cur_col_u=col_u[i]
            cur_col_u=mx.sym.BlockGrad(data=cur_col_u)
            mu_state=GRUState(h=m_u)
            next_state=myGRU(num_embed, indata=cur_col_u, prev_state=mu_state, param=param_ucells[0], seqidx=0, layeridx=0, dropout=dropout)
            m_u=next_state.h


            
        for i in xrange(nipass):
            cur_col_i=col_i[i]
            cur_col_i=mx.sym.BlockGrad(data=cur_col_i)
            mi_state=GRUState(h=m_i)
            next_state=myGRU(num_embed, indata=cur_col_i,prev_state=mi_state, param=param_icells[0], seqidx=0, layeridx=0, dropout=dropout)
            m_i=next_state.h
    m_u=mx.sym.Concat(m_u, mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_embed, weight=weight_emu), dim=1)
    m_i=mx.sym.Concat(m_i, mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_embed, weight=weight_emi), dim=1)
    pred=m_u*m_i
    pred=mx.sym.sum_axis(data=pred, axis=1)
    pred=mx.sym.Flatten(data=pred)
    pred=mx.sym.LinearRegressionOutput(data=pred, label=rating)
#    pred=mx.sym.FullyConnected(data=pred, num_hidden=5, name='cls')   
#    pred=mx.sym.SoftmaxOutput(data=pred, label=rating)
    return pred








def get_c_attention_nn(batch_size, num_embed, num_hidden, num_layer, num_user, num_item, nupass, nipass, npass, dropout=0.):
    last_states=[]
    param_ucells=[]
    param_icells=[]
    param_cells=[]
    for i in xrange(num_layer):
        param_ucells.append(GRUParam(gates_i2h_weight=mx.sym.Variable('ul%d_i2h_gates_weight'%i), gates_i2h_bias=mx.sym.Variable('ul%d_i2h_gates_bias'%i),gates_h2h_weight=mx.sym.Variable('ul%d_h2h_gates_weight'%i),gates_h2h_bias=mx.sym.Variable('ul%d_h2h_gates_bias'%i),trans_i2h_weight=mx.sym.Variable('ul%d_i2h_trans_weight'%i), trans_i2h_bias=mx.sym.Variable('ul%d_i2h_bias'%i), trans_h2h_weight=mx.sym.Variable('ul%d_h2h_trans_weight'%i), trans_h2h_bias=mx.sym.Variable('ul%d_h2h_bias'%i)))
        param_icells.append(GRUParam(gates_i2h_weight=mx.sym.Variable('il%d_i2h_gates_weight'%i), gates_i2h_bias=mx.sym.Variable('il%d_i2h_gates_bias'%i),gates_h2h_weight=mx.sym.Variable('il%d_h2h_gates_weight'%i),gates_h2h_bias=mx.sym.Variable('il%d_h2h_gates_bias'%i),trans_i2h_weight=mx.sym.Variable('il%d_i2h_trans_weight'%i), trans_i2h_bias=mx.sym.Variable('il%d_i2h_bias'%i), trans_h2h_weight=mx.sym.Variable('il%d_h2h_trans_weight'%i), trans_h2h_bias=mx.sym.Variable('il%d_h2h_bias'%i)))
        state=GRUState(h=mx.sym.Variable('l%d_init_h'%i))
        last_states.append(state)
    user=mx.sym.Variable('user')
    item=mx.sym.Variable('item')
    rating=mx.sym.Variable('rating')
    grp_u=mx.sym.Variable('grp_u')
    grp_i=mx.sym.Variable('grp_i')
    
    weight_emu=mx.sym.Variable('emu_weight')
    weight_emi=mx.sym.Variable('emi_weight')

    weight_u=mx.sym.Variable('u_weight')
    weight_i=mx.sym.Variable('i_weight')

    #weight_uu=mx.sym.Variable('aff_u', shape=(num_embed, num_embed), init=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=2))
    #weight_ii=mx.sym.Variable('aff_i', shape=(num_embed, num_embed), init=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=2))

    weight_z1=mx.sym.Variable('z1_weight')
    weight_z2=mx.sym.Variable('z2_weight')
    weight_z3=mx.sym.Variable('z3_weight')
    bias_z1=mx.sym.Variable('z1_bias')
    bias_z2=mx.sym.Variable('z2_bias')
    bias_z3=mx.sym.Variable('z3_bias')
    m_u=mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_embed, weight=weight_u)
    m_i=mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_embed, weight=weight_i)

    grp_u= mx.sym.Embedding(data=grp_u, input_dim=num_user, output_dim=num_embed, weight=weight_u)
    grp_i= mx.sym.Embedding(data=grp_i, input_dim=num_item, output_dim=num_embed, weight=weight_i)

    col_u=mx.sym.SliceChannel(data=grp_u, axis=1, num_outputs=nupass, squeeze_axis=1)
    col_i=mx.sym.SliceChannel(data=grp_i, axis=1, num_outputs=nipass, squeeze_axis=1)
    
    for _ in xrange(npass):
        for i in xrange(nupass):
            cur_col_u=col_u[i]
            #cur_col_u=mx.sym.BlockGrad(data=cur_col_u)
            q_u=mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_embed, weight=weight_u)
            z=[cur_col_u, m_u, q_u,  mx.sym.abs(q_u-cur_col_u), mx.sym.abs(cur_col_u-m_u)]
            #z=[ mx.sym.abs(q_u-m_u), mx.sym.abs(q_u-cur_col_u), mx.sym.abs(cur_col_u-m_u)]
            z=mx.sym.Concat(*z, dim=1)
            z=mx.sym.FullyConnected(data=z, num_hidden=num_hidden,weight=weight_z1, bias=bias_z1)
            z=mx.sym.Activation(data=z, act_type='relu')
            z=mx.sym.BatchNorm(data=z, fix_gamma=True)
            z=mx.sym.FullyConnected(data=z, num_hidden=nupass, weight=weight_z2, bias=bias_z2)
            z=mx.sym.Activation(data=z, act_type='sigmoid')
            z=mx.sym.FullyConnected(data=z, num_hidden=num_embed, weight=weight_z3, bias=bias_z3)
            gu=mx.sym.SoftmaxActivation(data=z)
            mu_state=GRUState(h=m_u)
            next_state=myGRU(num_embed, indata=cur_col_u, prev_state=mu_state, param=param_ucells[0], seqidx=0, layeridx=0, dropout=dropout)
            m_u=(next_state.h-m_u)*gu+m_u
            
        for i in xrange(nipass):
            cur_col_i=col_i[i]
            #cur_col_i=mx.sym.BlockGrad(data=cur_col_i)
            q_i=mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_embed, weight=weight_i)
            z=[cur_col_i, m_i, q_i, mx.sym.abs(q_i-cur_col_i), mx.sym.abs(cur_col_i-m_i)]
            #z=[mx.sym.abs(q_i-m_i), mx.sym.abs(q_i-cur_col_i), mx.sym.abs(cur_col_i-m_i)]
            z=mx.sym.Concat(*z, dim=1)
            z=mx.sym.FullyConnected(data=z, num_hidden=num_hidden,weight=weight_z1, bias=bias_z1)
            z=mx.sym.Activation(data=z, act_type='relu')
            z=mx.sym.BatchNorm(data=z, fix_gamma=True)
            z=mx.sym.FullyConnected(data=z, num_hidden=nipass, weight=weight_z2, bias=bias_z2)
            z=mx.sym.Activation(data=z, act_type='sigmoid')
            z=mx.sym.FullyConnected(data=z, num_hidden=nipass, weight=weight_z3, bias=bias_z3)
            gi=mx.sym.SoftmaxActivation(data=z)
            mi_state=GRUState(h=m_i)
            next_state=myGRU(num_embed, indata=cur_col_i,prev_state=mi_state, param=param_icells[0], seqidx=0, layeridx=0, dropout=dropout)
            m_i=(next_state.h-m_i)*gu+m_i
    #m_u=mx.sym.Concat(m_u, mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_embed, weight=weight_emu), dim=1)
    #m_i=mx.sym.Concat(m_i, mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_embed, weight=weight_emi), dim=1)
    
    pred=m_u*m_i
    pred=mx.sym.sum_axis(data=pred, axis=1)
    pred=mx.sym.Flatten(data=pred)
    pred=mx.sym.LinearRegressionOutput(data=pred, label=rating)
#    pred=mx.sym.FullyConnected(data=pred, num_hidden=5, name='cls')   
#    pred=mx.sym.SoftmaxOutput(data=pred, label=rating)
    return pred

