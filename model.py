import re

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.gradproc import ScaleGradient, SummaryGradient
from tensorpack.tfutils import optimizer

import ops


INPUT_SHAPE = 32


def get_weights(regex):
    result = list()

    for op in tf.trainable_variables():
        if re.search(regex, op.name):
            result.append(op)

    return result


def feed_forward(l, fancy, num_classes=10, growth=12, net=None):
    net = net or dict()

    layer = ops.look_ahead if fancy else ops.conv

    l = layer('conv1_1', l, growth, 11, net=net)
    l = layer('conv1_2', l, growth, 11, stride=2, net=net)

    l = layer('conv2_1', l, growth * 2, 5, net=net)
    l = layer('conv2_2', l, growth * 2, 5, stride=2, net=net)

    l = layer('conv3_1', l, growth * 4, 3, net=net)
    l = layer('conv3_2', l, growth * 4, 3, net=net)

    l = GlobalAvgPooling('pool', l)
    l = tf.expand_dims(l, 1)
    l = tf.expand_dims(l, 1)

    l = layer('fc_1', l, growth * 8, 1, net=net)
    l = layer('fc_2', l, growth * 8, 1, net=net)

    l = tf.squeeze(l, 1)
    l = tf.squeeze(l, 1)
    l = ops.fully_connected('logits', l, num_classes, net=net)

    return l, net


def get_loss(labels, logits, scope, alpha=5e-5):
    with tf.name_scope('loss'):
        xent_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits), name='xent')
        reg_loss = tf.multiply(
                alpha, regularize_cost('%s/.*/W' % scope, tf.nn.l2_loss), name='reg')
        total_loss = tf.add(xent_loss, reg_loss, name='total')

        # Tensorboard
        add_moving_summary(xent_loss)
        add_moving_summary(reg_loss)
        add_moving_summary(total_loss)

    with tf.name_scope('metrics'):
        prediction = tf.cast(tf.argmax(logits, -1), tf.int32)
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(prediction, labels)), name='accuracy')

    return total_loss


def get_kmeans_loss(net):
    with tf.name_scope('loss'):
        kmeans_loss = 0.0

        for name, activations in net.items():
            if 'conv' in name:
                kmeans_loss += -tf.reduce_mean(tf.reduce_max(activations, -1))

        loss = kmeans_loss

        kmeans_loss = tf.identity(kmeans_loss, 'kmeans')
        loss = tf.identity(loss, 'initialize')

        add_moving_summary(kmeans_loss)
        add_moving_summary(loss)

    return loss


class Network(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs

        tf.summary.image('image/input', tf.cast(image, tf.uint8), 5)

        # Vanilla.
        with tf.variable_scope('Vanilla'):
            logits_vanilla, _ = feed_forward(image, False)
            loss_vanilla = get_loss(label, logits_vanilla, 'Vanilla')

        # # Fancy.
        # with tf.variable_scope('Fancy'):
        #     logits_fancy, _ = feed_forward(image, True)
        #     loss_fancy = get_loss(label, logits_fancy, 'Fancy')

        # self.cost = loss_vanilla + loss_fancy
        self.cost = loss_vanilla

    def _get_optimizer(self):
        learn_rate_op = tf.Variable(0.0, trainable=False, name='learning_rate')
        optimizer_op = tf.train.AdamOptimizer(learn_rate_op)

        tf.summary.scalar('learn_rate', learn_rate_op)

        return optimizer_op


class Pretrain(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input')]

    def _build_graph(self, inputs):
        image = inputs[0]

        with tf.variable_scope('Vanilla'):
            _, net = feed_forward(image, False)
            kmeans_loss = get_kmeans_loss(net)

        self.weights = get_weights('Vanilla/conv1_1/W') + get_weights('Vanilla/conv1_2/W')
        self.net = net
        self.cost = kmeans_loss

        # add_param_summary(('.*', ['histogram', 'rms']))

    def _get_optimizer(self):
        learn_rate_op = tf.Variable(0.0, trainable=False, name='learning_rate')
        optimizer_op = tf.train.MomentumOptimizer(learn_rate_op, 0.9)

        tf.summary.scalar('learn_rate', learn_rate_op)

        return optimizer.apply_grad_processors(
                optimizer_op,
                [ScaleGradient([('.*/b', 0.0), ('.*/beta', 0.0), ('.*/gamma', 0.0)]),
                 SummaryGradient()])
