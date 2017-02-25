import mxnet as mx
import numpy as np
import logging, model



class AutoEncoderModel(model.mxmodel):
    def setup(self, dims, sparseness_penalty=None, pt_dropout=None, ft_dropout=None, input_act=None, internal_act='relu', output_act=None):
        self.N=len(dims)-1
        self.dims=dims
        self.stacks=[]
        self.pt_dropout=pt_dropout
        self.ft_dropout=ft_dropout
        self.input_act=input_act
        self.internal_act=internal_act
        self.out_act=output_act

        self.data=mx.sym.Variable('data')

        

        #Simple encoder and decoder
        self.encoder, self.internals=self.make_encoder(self.data, dims, sparseness_penalty, ft_dropout, internal_act, output_act)
        self.decoder = self.make_decoder(self.encoder, dims, sparseness_penalty, ft_dropout, internal_act, input_act)
        if input_act=='softmax':
            self.loss=self.decoder
        else:
            self.loss=mx.sym.LinearRegressionOutput(data=self.decoder, label=self.data)

    def make_encoder(self, data, dims, sparseness_penalty=None,dropout=None, internal_act='relu', output_act=None):
        x=data
        internals=[]
        N=len(dims)-1
        for i in xrange(N):
            x=mx.sym.FullyConnected(data=x, name='encoder_%d'%i, num_hidden=dims[i+1])
            if internal_act=='sigmoid' and sparseness_penalty:
                x=mx.sym.IdentityAttachKLSparseReg(data=x, name='sparse_encoder_%d'%i, penalty=sparseness_penalty)
            elif output_act and i==N-1:
                x=mx.sym.Activation(data=x, act_type=output_act)
                if output_act=='sigmoid' and sparseness_penalty:
                    x=mx.sym.IdentityAttachKLSparseReg(data=x, name='sparse_encoder_%d'%i, penalty=sparseness_penalty)
            if dropout:
                x=mx.sym.Dropout(data=x,p=dropout)
            internals.append(x)
        return x, internals

    def make_decoder(self, feature, dims, sparseness_penalty=None, dropout=None, internal_act='relu', input_act=None):
        x=features
        N=len(dims)-1
        for i in reversed(range(N)):
            x=mx.sym.FullyConnected(data=x, name='decoder_%d'%i,num_hidden=dims[i])
            if internal_act and i:
                x=mx.sym.Activation(data=x, act_type=internal_act)
                if internal_act=='sigmoid' and sparseness_penalty:
                    x=mx.sym.IdentityAttachKLSparseReg(data=x, name='sparse_decoder_%d'%i, penalty=sparseness_penalty)
            elif input_act and i==0:
                x=mx.sym.Activation(data=x, act_type=input_act)
                if input_act=='sigmoid' and sparseness_penalty:
                    x=mx.sym.IdentityAttachKLSparseReg(data=x, name='sparse_decoder_%d'%i, penalty=sparseness_penalty)
            if dropout and i:
                x=mx.sym.Dropout(data=x, p=dropout)
        return x

    def eval(self, x):
        batch_size=100
        data_iter=mx.io.NDArrayIter({'data':x}, batch_size=batch_size, shuffle=False, last_batch_handle='pad')
        Y=model.extract_feature(self.loss, self.args, self.auxs, data_iter, X.shape[0], self.xpu).values()[0]
        return np.mean(np.square(Y-x))/2.0
