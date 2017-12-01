import tensorflow as tf
import numpy as np

from tensorpack import *


def look_ahead(name, l, channel, k=3, stride=1, net=None):
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
        net[name] = tf.identity(l, name)

    return l


def conv(name, l, channel, k=3, stride=1, net=None):
    with tf.variable_scope(name):
        l = BNReLU('bn_relu', l)
        l = vanilla_conv(l, channel, k, stride, True)

        # l = Conv2D('conv', l, channel, k, stride=stride)

    if net is not None:
        net[name] = tf.identity(l, name)

    return l


def kernel_norm(W, eps=1e-9):
    return tf.sqrt(tf.reduce_sum(W * W, (0, 1, 2)) + eps)


def vanilla_conv(l, channel, k, stride, normalize):
    W = tf.get_variable('W', [k, k, l.shape.as_list()[-1], channel], tf.float32,
            initializer=tf.random_normal_initializer(stddev=1e-3))
    b = tf.get_variable('b', [channel], tf.float32,
            initializer=tf.constant_initializer(0.0))

    if normalize:
        W = W / kernel_norm(W)

    return tf.nn.conv2d(l, W, [1, stride, stride, 1], padding='SAME') + b


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
        # networks = ['Vanilla', 'Fancy']
        networks = ['Vanilla']
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


class InitializationCallback(Callback):
    def __init__(self, data, model):
        self.data = data
        self.model = model

        self.values = list()
        self.bad_count = 0

    def _setup_graph(self):
        networks = ['Vanilla']
        tensors = ['loss/kmeans', 'loss/initialize']
        names = ['conv1_1_1', 'conv2_1_1', 'conv3_1_1']

        self.monitor = ['%s/%s' % (x, y) for x in networks for y in tensors]
        self.scalars = self.trainer.get_predictor(['input'], self.monitor)

        self.names = ['%s/%s' % (x, y) for x in networks for y in names]
        self.activations = self.trainer.get_predictor(['input'], self.names)

    def _before_train(self):
        self.data.reset_state()

    def _trigger(self):
        epsilon = 1.0
        num_images = 3
        num_channels = 3

        # Scalar metrics.
        metrics = {metric: list() for metric in self.monitor}

        for image in self.data.get_data():
            for monitor, scalars in zip(self.monitor, self.scalars(image[0])):
                metrics[monitor].append(scalars)

        for metric_name in metrics:
            metrics[metric_name] = np.mean(metrics[metric_name])

        for metric_name, metric in metrics.items():
            self.trainer.monitors.put_scalar('val/%s' % metric_name, metric)

        # Activations.
        for batch_number, image in enumerate(self.data.get_data()):
            for name, activations in zip(self.names, self.activations(image[0])):
                for image_number in range(num_images):
                    self.trainer.monitors.put_image(
                            'input/%s' % image_number, image[0][image_number])

                    for channel_number in range(num_channels):
                        name = '%s/%s/%s' % (name, image_number, channel_number)
                        activation = activations[image_number,:,:,channel_number]
                        activation = (activation - np.min(activation)) / (np.max(activation) - np.min(activation))
                        activation = np.uint8(255.0 * activation)

                        self.trainer.monitors.put_image(name, activation)

            if batch_number == 0:
                break

        # Weights.
        for weights_op in self.model.weights:
            weights_name = ''.join(weights_op.name.split('/')[1:])

            weights = weights_op.eval()
            weights_norm = np.sqrt(np.sum(weights * weights, (0, 1, 2)) + 1e-9)
            weights = weights / weights_norm

            # Normalize each kernel.
            weights_op.load(weights)

            for i in range(weights.shape[-1]):
                weight = weights[:,:,:,i]
                weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
                weight = np.uint8(255.0 * weight)

                self.trainer.monitors.put_image('%s/%s' % (weights_name, i), weight)

        # Stopping criteria.
        self.values.append(metrics['Vanilla/loss/kmeans'])

        if len(self.values) >= 2 and self.values[-2] - self.values[-1] < epsilon:
            self.bad_count += 1

            if self.bad_count == 3:
                raise StopTraining()
        else:
            self.bad_count = 0
