import os
import logging
from nowcasting.movingmnist_iterator import MovingMNISTIterator
from nowcasting.operators import *
from nowcasting.ops import *
from nowcasting.helpers.gifmaker import save_gif
from nowcasting.utils import norm_clipping, cross_entropy_npy



class MovingMNISTFactory(object):
    def __init__(self, batch_size=32, in_seq_len=10, out_seq_len=10,
                 conv_rnn_typ="ConvGRU",
                 use_ss=True,
                 transform_typ="direct"):
        self._conv_rnn_typ = conv_rnn_typ
        assert transform_typ in ["direct", "CDNA", "DFN"]
        self._transform_typ = transform_typ
        self._batch_size = batch_size
        self._in_seq_len = in_seq_len
        self._out_seq_len = out_seq_len
        if self._conv_rnn_typ == "DFNConvRNN":
            self.conv_rnn1 = DFNConvRNN(num_filter=128, h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                                        i2h_kernel=(3, 3), i2h_pad=(1, 1), #act_type="leaky",
                                        name="conv_rnn1")
            self.conv_rnn1_states = None
        elif self._conv_rnn_typ == "ConvGRU":
            self.conv_rnn1 = ConvGRU(num_filter=128, h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                                     i2h_kernel=(3, 3), i2h_pad=(1, 1),  act_type="tanh",
                                     name="conv_rnn1")
            self.conv_rnn1_states = None
        elif self._conv_rnn_typ == "ConvGRU_leaky":
            self.conv_rnn1 = ConvGRU(num_filter=128, h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                                     i2h_kernel=(3, 3), i2h_pad=(1, 1), act_type="leaky",
                                     name="conv_rnn1")
            self.conv_rnn1_states = None
        elif self._conv_rnn_typ == "ConvGRUI_leaky":
            self.conv_rnn1 = ConvGRUI(num_filter=128, h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                                     i2h_kernel=(3, 3), i2h_pad=(1, 1), act_type="leaky",
                                     name="conv_rnn1")
            self.conv_rnn1_states = None
        else:
            raise NotImplementedError

    def reset_all(self):
        reset_regs()
        self.conv_rnn1.reset_states()

    def train_sym(self):
        self.reset_all()
        data = mx.sym.Variable('data')  # Shape: (in_seq_len, batch_size, C, H, W)
        target = mx.sym.Variable('target')  # Shape: (out_seq_len, batch_size, C, H, W)
        gt_prob = mx.sym.Variable('gt_prob')
        forecast_target = self.gen_prediction(data=data,
                                              gt=None,
                                              gt_prob=None,
                                              use_ss=False,
                                              batch_size=self._batch_size,
                                              in_seq_len=self._in_seq_len,
                                              out_seq_len=self._out_seq_len)
        forecast_target = mx.sym.clip(forecast_target, a_min=1E-6, a_max=1 - 1E-6)
        cross_entropy_loss = mx.sym.MakeLoss((- target * mx.sym.log(forecast_target)
                                             - (1-target) * mx.sym.log(1 - forecast_target))/self._out_seq_len,
                                             grad_scale=1.0)
        # squared_loss = mx.sym.MakeLoss(mx.sym.square(target - forecast_target),
        #                                grad_scale=1.0)
        return cross_entropy_loss

    def test_sym(self):
        self.reset_all()
        data = mx.sym.Variable('data')  # Shape: (in_seq_len, batch_size, C, H, W)
        forecast_target = self.gen_prediction(data=data,
                                              batch_size=self._batch_size,
                                              in_seq_len=self._in_seq_len,
                                              out_seq_len=self._out_seq_len)
        return forecast_target

    def gen_prediction(self, data, batch_size, in_seq_len, out_seq_len, gt=None, gt_prob=None,
                       use_ss=False):
        if use_ss:
            gt = mx.sym.SliceChannel(gt, num_outputs=out_seq_len, axis=0, squeeze_axis=False)
        # Encoder
        states = self.encoder(data=data, seqlen=in_seq_len)
        # FNN Forecasting
        I_tm1 = mx.sym.slice_axis(data=data, axis=0, begin=in_seq_len - 1,
                                  end=in_seq_len)
        I_tm1 = mx.sym.Reshape(I_tm1, shape=(0, 0, 0, 0), reverse=True)
        forecast_target = self.one_step_forecast(states=states, I_tm1=I_tm1)

        I_tm1 = forecast_target
        forecast_target = mx.sym.Reshape(forecast_target, shape=(1, 0, 0, 0, 0), reverse=True)
        if out_seq_len > 1:
            forecast_vec = []
            forecast_vec.append(forecast_target)
            for i in range(out_seq_len - 1):
                if use_ss:
                    # Use scheduled sampling to choose whether to use GT or Generated sample
                    forecast_target = SS(gt=gt[i], pred=forecast_target, gt_prob=gt_prob,
                                         batch_size=batch_size)
                    # forecast_target = gt[i]
                states = self.encoder(data=forecast_target, seqlen=1)
                forecast_target = self.one_step_forecast(states=states, I_tm1=I_tm1)
                I_tm1 = forecast_target
                forecast_target = mx.sym.Reshape(forecast_target, shape=(1, 0, 0, 0, 0),
                                                 reverse=True)
                forecast_vec.append(forecast_target)
            forecast_target = mx.sym.Concat(*forecast_vec, num_args=len(forecast_vec), dim=0)
            return forecast_target
        else:
            return forecast_target

    def encoder(self, data, seqlen=1):
        data = mx.sym.Reshape(data, shape=(-1, 0, 0, 0), reverse=True)
        conv1 = conv2d_act(data=data, num_filter=32, kernel=(5, 5), stride=(1, 1),
                           pad=(2, 2),
                           act_type="leaky",
                           name="conv1")
        conv2 = conv2d_act(data=conv1, num_filter=32, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                           act_type="leaky",
                           name="conv2")
        conv3 = conv2d_act(data=conv2, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                           act_type="leaky",
                           name="conv3")
        conv4 = conv2d_act(data=conv3, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                           act_type="leaky",
                           name="conv4")
        conv4 = mx.sym.Reshape(conv4, shape=(seqlen, -1, 0, 0, 0), reverse=True)
        conv4_sliced = mx.sym.SliceChannel(conv4,
                                           num_outputs=seqlen,
                                           axis=0, squeeze_axis=True)
        conv_rnn1_states = self.conv_rnn1.get_states()
        for i in range(seqlen):
            conv_rnn1_out, conv_rnn1_states = self.conv_rnn1.step(inputs=conv4_sliced[i],
                                                                  states=conv_rnn1_states)
        return conv4_sliced[seqlen - 1], conv_rnn1_out

    def one_step_forecast(self, states, I_tm1=None):
        conv4, conv_rnn1_out = states
        state_in = mx.sym.Concat(conv4, conv_rnn1_out, num_args=2, dim=1)
        conv5 = conv2d_act(data=state_in, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                           act_type="leaky",
                           name="conv5")
        conv6 = conv2d_act(data=conv5, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                           act_type="leaky",
                           name="conv6")
        # up7 = mx.sym.UpSampling(conv6, scale=2, sample_type='nearest', num_args=1)
        deconv7 = deconv2d_act(data=conv6, num_filter=64, kernel=(4, 4), stride=(2, 2),
                               pad=(1, 1),
                               act_type="leaky",
                               name="deconv7")
        # deconv7 = identity(deconv7, name="deconv7", input_debug=True, grad_debug=True)
        conv8 = conv2d_act(data=deconv7, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                           act_type="leaky",
                           name="conv8")
        conv9 = conv2d_act(data=conv8, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                           act_type="leaky",
                           name="conv9")
        # conv9 = identity(conv9, name="conv9", input_debug=True, grad_debug=True)
        conv10 = conv2d_act(data=conv9, num_filter=128, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                            act_type="leaky",
                            name="conv10")
        if self._transform_typ == "direct":
            conv11 = conv2d_act(data=conv10, num_filter=1, kernel=(1, 1), stride=(1, 1),
                               pad=(0, 0), name="conv11", act_type="sigmoid")
            return conv11
        elif self._transform_typ == "CDNA":
            mask = conv2d(data=conv10, num_filter=5, kernel=(1, 1), stride=(1, 1),
                          pad=(0, 0), name="kernels")
            kernels = fc_layer(data=conv5, num_hidden=5 * 5 * 5, name="mask")
            pred = CDNA(data=I_tm1, kernels=kernels, mask=mask, batch_size=self._batch_size,
                        num_filter=5, kernel_size=5)
            return pred
        elif self._transform_typ == "DFN":
            # conv10 = identity(conv10, name="conv10", input_debug=True, grad_debug=True)
            local_kernels = conv2d_act(data=conv10, num_filter=9*9, kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0), name="local_kernels", act_type="identity")
            pred = DFN(data=I_tm1, local_kernels=local_kernels, K=9, batch_size=self._batch_size)
            return pred

batch_size = 8
in_seq_len = 10
out_seq_len = 10
base_dir = "MovingMNIST"
ctx = mx.gpu()
logging_config(folder=base_dir)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
transform_typ = "DFN"
conv_rnn_typ = "DFNConvRNN"
moving_mnist_factory = MovingMNISTFactory(batch_size=batch_size, in_seq_len=in_seq_len, out_seq_len=out_seq_len,
                                          conv_rnn_typ=conv_rnn_typ, transform_typ=transform_typ)
train_sym = moving_mnist_factory.train_sym()
test_sym = moving_mnist_factory.test_sym()
net = mx.mod.Module(train_sym,
                    data_names=('data',), label_names=('target',), context=ctx)
test_net = mx.mod.Module(test_sym,
                         data_names=('data',), label_names=None, context=ctx)
net.bind(data_shapes=[mx.io.DataDesc(name='data', shape=(in_seq_len, batch_size, 1, 64, 64),
                                     layout="SNCHW")],
         label_shapes=[mx.io.DataDesc(name='target', shape=(out_seq_len, batch_size, 1, 64, 64),
                                      layout="SNCHW")],
         grad_req='write')
init = mx.init.Xavier(factor_type="in", magnitude=1)
net.init_params(initializer=init)
net.init_optimizer(optimizer='adam',
                   kvstore=None,
                   optimizer_params={'learning_rate': 1E-3,
                                     'wd': 1E-5})
test_net.bind(data_shapes=[('data', (in_seq_len, batch_size, 1, 64, 64))],
              label_shapes=None, for_training=False,
              grad_req='null', shared_module=net)


mnist_iter = MovingMNISTIterator()

for i in range(20000):
    seq = mnist_iter.sample(digitnum=2, width=64, height=64, seqlen=in_seq_len+out_seq_len,
                            batch_size=batch_size, lower=3.6, upper=3.6)
    in_seq = seq[:in_seq_len, ...]
    gt_seq = seq[in_seq_len:(in_seq_len + out_seq_len), ...]
    net.forward(data_batch=mx.io.DataBatch(data=[mx.nd.array(in_seq)/255.0],
                                           label=[mx.nd.array(gt_seq)/255.0]),
                is_train=True)
    outputs = net.get_outputs()
    net.backward()
    norm_val = get_global_norm_val(net)
    # norm_clipping(params_grad=[grad[0] for grad in net._exec_group.grad_arrays],
    #               threshold=100, batch_size=batch_size)
    logging.info("Iter:%d, Error:%f, Norm:%f" %(i,
                                                outputs[0].asnumpy().sum()/batch_size/64/64,
                                                norm_val))

    for k, v, grad_v in zip(net._param_names, net._exec_group.param_arrays, net._exec_group.grad_arrays):
        if "bn" not in k:
            print k, v[0].shape, nd.norm(v[0]).asnumpy(), nd.norm(grad_v[0]/batch_size).asnumpy()
    net.update()
    if (i + 1) % 100 == 0:
        test_net.forward(data_batch=mx.io.DataBatch(data=[mx.nd.array(in_seq) / 255.0],
                                                    label=None),
                         is_train=False)
        test_prediction = test_net.get_outputs()[0].asnumpy()
        logging.info("Iter:%d, Test Error:%f" % (i,
                                                 -cross_entropy_npy(gt_seq/255.0, test_prediction).sum()/out_seq_len/batch_size))
        save_gif(test_prediction[:, 0, 0, :, :], "test.gif")
    if (i + 1) % 2000 == 0:
        net.save_checkpoint(prefix=os.path.join(base_dir, "%s_%s" %(conv_rnn_typ, transform_typ)),
                            epoch=i)
