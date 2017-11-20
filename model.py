import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.gradproc import SummaryGradient

import ops


INPUT_SHAPE = 32


def feed_forward(l, scope, num_classes=10, growth=16, net=None):
    net = net or dict()

    with tf.variable_scope(scope):
        l = ((l / 255.0) - 0.5) * 2.0

        l = ops.conv('conv1_1', l, growth, net=net)
        l = ops.conv('conv1_2', l, growth, stride=2, net=net)

        l = ops.conv('conv2_1', l, growth * 2, net=net)
        l = ops.conv('conv2_2', l, growth * 2, stride=2, net=net)

        l = ops.conv('conv3_1', l, growth * 4, net=net)
        l = ops.conv('conv3_2', l, growth * 4, stride=2, net=net)

        l = ops.conv('conv4_1', l, growth * 8, net=net)
        l = ops.conv('conv4_2', l, growth * 8, stride=2, net=net)

        l = AvgPooling('pool', l, 2)
        l = ops.fully_connected('fc1', l, 256, net)
        l = ops.fully_connected('fc2', l, 256, net)
        l = ops.fully_connected('logits', l, num_classes, net)

    return l, net


def get_loss(labels, logits, scope, alpha=5e-5):
    with tf.name_scope('%s/loss' % scope):
        xent_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits), name='xent')
        reg_loss = tf.multiply(
                alpha, regularize_cost('.*/W', tf.nn.l2_loss), name='reg')
        total_loss = tf.add(xent_loss, reg_loss, name='total')

        # Tensorboard
        add_moving_summary(xent_loss)
        add_moving_summary(reg_loss)
        add_moving_summary(total_loss)

    return total_loss


class Network(ModelDesc):
    def __init__(self, scope):
        super().__init__()

        self.scope = scope

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input1'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        logits, _ = feed_forward(image, self.scope)
        loss = get_loss(label, logits, self.scope)

        tf.summary.image('image/input', tf.cast(image, tf.uint8), 5)

        self.cost = loss

    def _get_optimizer(self):
        learn_rate_op = tf.Variable(1e-4, trainable=False, name='learning_rate')
        optimizer_op = tf.train.AdamOptimizer(learn_rate_op)

        tf.summary.scalar('learn rate', learn_rate_op)

        return optimizer_op
