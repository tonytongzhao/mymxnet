import mxnet as mx
import numpy as np



def get_memnn(edim, mem_size, nwords, nhops, lindim):
    data=mx.sym.Variable('data')
    time=mx.sym.Variable('time')
    target=mx.sym.Variable('target')
    context=mx.sym.Variable('context')

    hid=[]
    hid.append(data)

    share_list=[]
    share_list.append([])

    A=mx.sym.Variable('A_emb', shape=(nwords, edim))
    C=mx.sym.Variable('C_emb', shape=(nwords, edim))
    H=mx.sym.Variable('H'. shape=(edim,edim))


    T_A=mx.sym.Variable('TA_weight', shape=(mem_size, edim))
    T_C=mx.sym.Variable('TC_weight', shape=(mem_size, edim))

    #Embedding contexts and temporal factor
    a_embed=mx.sym.Embedding(data=context, input_dim=nwords, output_dim=edim, weight=A)
    at_embed=mx.sym.Embedding(data=time, input_dim=mem_size, output_dim=edim, weight=T_A)
    a_embd=a_embed+at_embed

    c_embed=mx.sym.Embedding(data=context, input_dim=nwords, output_dim=edim, weight=C)
    ct_embed=mx.sym.Embedding(data=time, input_dim=mem_size, output_dim=edim, weight=T_C)
    c_embd=c_embed+ct_embed

    for h in xrange(nhops):
        #Calculate p
        p=mx.sym.batch_dot(lhs=mx.sym.expand_dims(hid[-1], axis=1), rhs=mx.sym.transpose(a_embd, axes=(0,2,1)))
        p=mx.sym.Flatten(p)
        p=mx.sym.SoftmaxActivation(p)

        #Calcluate o
        o=mx.sym.batch_dot(lhs=mx.sym.expand_dims(p, axis=1), rhs=c_embd)
        o=mx.sym.Flatten(o)

        #Calculate mu
        trans_mu=mx.sym.doth(hid[-1], H)
        mu=trans_mu+o

        share_list[0].append(trans_mu)

        if lindim==edim:
            hid.append(mu)
        elif lindim==0:
            hid.append(mx.sym.Activation(data=mu, act_type='relu'))

        else:
            f=mx.sym.Crop(mx.sym.Reshape(mu, shape=(-1,1,1,edim)), num_args=1, offset=(0,0),h_w=(1,limdim))
            f=mx.sym.Reshape(f, shape=(-1,lindim))
            
            g=mx.sym.Crop(mx.sym.Reshape(mu, shape=(-1,1,1,edim)), num_args=1, offset=(0,lindim),h_w=(1,edim-limdim))
            g=mx.sym.Reshape(f, shape=(-1,edim-lindim))
            g=mx.sym.Activation(data=g, act_type='relu')
            hid.append(mx.sym.Concat(f,g, dim=1))

        cls=mx.sym.FullyConnected(data=hid[-1], num_hidden=nwords, no_bias=True, name='cls') 
        loss=mx.sym.SoftmaxOutput(data=cls, label=target, name='prob')
        return loss
            
