import mxnet as mx
import numpy as np
import random
from collections import namedtuple


def get_ncf(batch_size, num_embed, num_hidden, num_layer, num_user, num_item,dropout=0.):
    user=mx.sym.Variable('user')
    item=mx.sym.Variable('item')
    rating=mx.sym.Variable('rating')

    mf_weight_u=mx.sym.Variable('mfu_weight')
    mf_weight_i=mx.sym.Variable('mfi_weight')

    mlp_weight_u=mx.sym.Variable('mlpu_weight')
    mlp_weight_i=mx.sym.Variable('mlpi_weight')

    #GMF Part
    mf_u=mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_embed, weight=mf_weight_u)
    mf_i=mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_embed, weight=mf_weight_i)

    mf_res=mf_u*mf_i


    #MLP Part
    mlp_u=mx.sym.Embedding(data=user, input_dim=num_user, output_dim=num_hidden[0], weight=mlp_weight_u)
    mlp_i=mx.sym.Embedding(data=item, input_dim=num_item, output_dim=num_hidden[0], weight=mlp_weight_i)
    mlp_data=mx.sym.Concat(*[mlp_u, mlp_i], dim=1)
    for i in xrange(1, num_layer+1):
        mlp_data=mx.sym.FullyConnected(data=mlp_data, weight=mx.sym.Variable('fc_%d_weight'%(i)), bias=mx.sym.Variable('fc_%d_bias'%(i)), num_hidden=num_hidden[i])
        mlp_data=mx.sym.Activation(data=mlp_data, act_type='relu')

    #Merge two parts
    pred=mx.sym.Concat(*[mlp_data, mf_res], dim=1)
    pred=mx.sym.FullyConnected(data=pred, num_hidden=1, name='cls')
    pred=mx.sym.LinearRegressionOutput(data=pred, label=rating)
    return pred


        







