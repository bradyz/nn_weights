import tensorflow as tf
import numpy as np

from tensorpack import *


def look_ahead(name, l, channel, k=3, stride=2, net=None):
    channel = channel // 3

    with tf.variable_scope(name):
        l = BNReLU('bn_relu', l)

        phi = Conv2D('conv_out', l, 2 * channel, k, stride=stride)
        phi_1 = phi[:,:,:,:channel]
        phi_2 = phi[:,:,:,channel:]

        alpha = Conv2D('conv_choose', l, channel, k, stride=stride)
        alpha = BatchNorm('bn_choose', alpha)
        alpha = tf.nn.sigmoid(alpha)

        l = alpha * phi_1 + (1.0 - alpha) * phi_2

    if net is not None:
        net[name] = l

    return l


def conv(name, l, channel, k=3, stride=1, net=None):
    with tf.variable_scope(name):
        l = BNReLU('bn_relu', l)
        l = Conv2D('conv', l, channel, k, stride=stride)

    if net is not None:
        net[name] = l

    return l


def fully_connected(name, l, channel, net=None):
    with tf.variable_scope(name):
        l = BNReLU('bn_relu', l)
        l = FullyConnected('fc', l, channel)

    if net is not None:
        net[name] = l

    return l


class ValidationCallback(Callback):
    def __init__(self, data, num_samples=10):
        self.data = data
        self.num_samples = num_samples

    def _setup_graph(self):
        networks = ['Vanilla', 'Fancy']
        tensors = ['loss/xent', 'metrics/accuracy']

        self.monitor = ['%s/%s' % (x, y) for x in networks for y in tensors]
        self.output = self.trainer.get_predictor(['input', 'label'], self.monitor)

    def _before_train(self):
        self.data.reset_state()

    def _trigger(self):
        metrics = {metric: list() for metric in self.monitor}

        for image, label in self.data.get_data():
            for monitor, output in zip(self.monitor, self.output(image, label)):
                metrics[monitor].append(output)

        for metric, metric_list in metrics.items():
            self.trainer.monitors.put_scalar('val/%s' % metric, np.mean(metric_list))
