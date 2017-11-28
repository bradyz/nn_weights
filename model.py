import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.gradproc import SummaryGradient

import ops


INPUT_SHAPE = 32


def feed_forward(l, scope, fancy, num_classes=10, growth=12, net=None):
    net = net or dict()

    layer = ops.conv if not fancy else ops.look_ahead

    with tf.variable_scope(scope):
        l = ((l / 255.0) - 0.5) * 2.0

        l = ops.conv('conv1_1', l, growth, net=net)
        l = ops.conv('conv1_2', l, growth, stride=2, net=net)

        l = ops.conv('conv2_1', l, growth * 2, net=net)
        l = ops.conv('conv2_2', l, growth * 2, stride=2, net=net)

        l = layer('conv3_1', l, growth * 4, net=net)
        l = layer('conv3_2', l, growth * 4, net=net)

        l = layer('conv4_1', l, growth * 8, net=net)
        l = layer('conv4_2', l, growth * 8, net=net)

        l = GlobalAvgPooling('pool', l)
        l = ops.fully_connected('fc1', l, 128, net)
        l = ops.fully_connected('fc2', l, 128, net)
        l = ops.fully_connected('logits', l, num_classes, net)

    return l, net


def get_loss(labels, logits, scope, alpha=5e-5):
    with tf.name_scope('%s/loss' % scope):
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

    with tf.name_scope('%s/metrics' % scope):
        prediction = tf.cast(tf.argmax(logits, -1), tf.int32)
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(prediction, labels)), name='accuracy')

    return total_loss


class Network(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs

        tf.summary.image('image/input', tf.cast(image, tf.uint8), 5)

        # Vanilla.
        logits_vanilla, _ = feed_forward(image, 'Vanilla', False)
        loss_vanilla = get_loss(label, logits_vanilla, 'Vanilla')

        # fancy.
        logits_fancy, _ = feed_forward(image, 'Fancy', True)
        loss_fancy = get_loss(label, logits_fancy, 'Fancy')

        self.cost = loss_vanilla + loss_fancy

    def _get_optimizer(self):
        learn_rate_op = tf.Variable(1e-4, trainable=False, name='learning_rate')
        optimizer_op = tf.train.AdamOptimizer(learn_rate_op)

        tf.summary.scalar('learn rate', learn_rate_op)

        return optimizer_op
