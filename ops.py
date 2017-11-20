import tensorflow as tf
import numpy as np

from tensorpack import *


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


class VisualizeTestSet(Callback):
    def __init__(self, data, num_samples=10):
        self.data = data
        self.num_samples = num_samples

    def _setup_graph(self):
        self.output = self.trainer.get_predictor(
                ['input1', 'input2', 'label'],
                ['pred/baseline', 'ae_loss', 'pred_pred', 'flow_label'])

    def _before_train(self):
        self.data.reset_state()

    def _trigger(self):
        baselines = list()
        losses = list()

        i = 0

        for input1, input2, label in self.data.get_data():
            baseline, loss, pred, label_viz = self.output(input1, input2, label)

            baselines.append(baseline)
            losses.append(loss)

            if i == 0:
                self.trainer.monitors.put_image('val_label', label_viz)
                self.trainer.monitors.put_image('val_output', pred)

            i += 1

        self.trainer.monitors.put_scalar('val/ae_loss', np.mean(losses))
        self.trainer.monitors.put_scalar('val/baseline', np.mean(baseline))
