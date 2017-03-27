import mxnet as mx

def ffn(num_layer, seq_len, input_size, num_hidden, num_embed, num_label, dropout=0.):
    embed_weight=mx.sym.Variable('embed_weight')
    cls_weight=mx.sym.Variable('cls_weight')
    cls_bias=mx.sym.Variable('cls_bias')

    data=mx.sym.Variable('data')
    hds=mx.sym.Embedding(data=data, weight=embed_weight, input_dim=input_size, output_dim=num_embed, name='ffn_embed')

    net=mx.sym.sum_axis(data=hds, axis=1)/seq_len
    
    for i in xrange(num_layer):
        net=mx.sym.FullyConnected(data=net, num_hidden=num_hidden, name='fc%d'%(i))
        net=mx.sym.Activation(data=net, act_type='relu', name='relu%d'%(i))

    fc=mx.sym.FullyConnected(data=net, weight=cls_weight, bias=cls_bias, num_hidden=num_label, name='ffn_cls')
    loss=mx.sym.LogisticRegressionOutput(data=fc, label=mx.sym.Variable('label'))    
    return loss 
