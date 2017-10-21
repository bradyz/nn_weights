import tensorflow as tf

import ops


class VanillaNetwork(object):
    def __init__(self, image_op, num_classes, is_training_op,
                 save_path=None, labels_op=None, scope=None):
        self.image_op = image_op
        self.labels_op = labels_op

        self.num_classes = num_classes
        self.is_training_op = is_training_op

        self.scope = scope or 'VanillaNet'

        # If save path None, will not load or save anything.
        self.save_path = save_path

        # To be populated.
        self.logits_op = None
        self.pred_op = None

        # Will be none if labels are not provided.
        self.loss_op = None
        self.losses = dict()

        # Populate missing attributes.
        self._forward()
        self._losses()

        self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def _forward(self):
        net = self.image_op

        with tf.variable_scope(self.scope):
            net = ops.down_block(net, 32, self.is_training_op, 'block1')
            net = ops.down_block(net, 32, self.is_training_op, 'block2')
            net = ops.down_block(net, 64, self.is_training_op, 'block3')
            net = ops.down_block(net, 64, self.is_training_op, 'block4')
            net = ops.flatten(net)
            net = ops.dense_block(net, 64, self.is_training_op, 'block5')
            net = ops.dense_block(net, 64, self.is_training_op, 'block6')
            net = ops.dense_block(net, self.num_classes, self.is_training_op, 'logits',
                    activation=False)

        with tf.name_scope('predictions'):
            pred_op = tf.cast(tf.argmax(tf.nn.softmax(net), axis=-1), tf.int32)

        # Actually populated.
        self.logits_op = net
        self.pred_op = pred_op

    def _losses(self, alpha=5e-5):
        # No labels means no losses.
        if self.labels_op is None:
            return

        with tf.name_scope('loss'):
            xent_loss_op = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.logits_op,
                        labels=self.labels_op))

            reg_loss_op = alpha * tf.reduce_sum(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            loss_op = xent_loss_op + reg_loss_op

        losses = dict()
        losses['xent'] = xent_loss_op
        losses['reg'] = reg_loss_op

        # Actually populated.
        self.loss_op = loss_op
        self.losses = losses

    def ready_up(self, sess):
        if self.save_path is None:
            print('No save path provided.')
            return

        weights_dict = {key.name: key for key in self.weights}
        saver = tf.train.Saver(weights_dict)

        try:
            saver.restore(sess, self.save_path)

            print('Loaded weights successfully.')
        except Exception as e:
            print(e)
            print('Failed to load weights. Reinitializing.')

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())


class Monitor(object):
    """
    Network must have the following attributes:
    - pred_op
    - logits_op
    """
    def __init__(self, network, trainer, log_dir):
        self.network = network
        self.trainer = trainer

        # Events file will be saved here.
        self.log_dir = log_dir

        # Initialized after session is created.
        self.sess = None
        self.summary_writer = None

        train = lambda: self._make_summary(True, 'train')
        valid = lambda: self._make_summary(False, 'valid')

        self.summary_op = tf.cond(network.is_training_op, train, valid)


    def ready_up(self, sess):
        self.sess = sess
        self.summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

    def checkpoint(self, step):
        train = self.sess.run(self.summary_op, {self.network.is_training_op: True})
        valid = self.sess.run(self.summary_op, {self.network.is_training_op: False})

        self.summary_writer.add_summary(train, step)
        self.summary_writer.add_summary(valid, step)

    def _make_summary(self, is_training, scope):
        result = list()

        if is_training:
            result += tf.get_collection(tf.GraphKeys.SUMMARIES)
            result += [tf.summary.scalar('learn_rate', self.trainer.learn_rate_op)]

        with tf.name_scope(scope):
            result += [tf.summary.image('images', self.network.images_op, 10)]

            result += [tf.summary.scalar('accuracy',
                tf.reduce_mean(
                    tf.to_float(
                        tf.equal(
                            self.network.pred_op,
                            self.network.labels_op))))]

            result += [tf.summary.scalar('accuracy_tf',
                tf.metrics.accuracy(self.network.pred_op, self.network.labels_op))]

            result += [tf.summary.image('confusion_matrix',
                ops.confusion_image(
                    self.network.pred_op, self.network.labels_op,
                    self.network.num_classes))]

        return tf.summary.merge(result)


class Trainer(object):
    """
    Network must have the following methods:
    - is_training_op
    - loss_op
    - weights
    """
    def __init__(self, network, scope=None):
        self.network = network

        self.scope = scope or 'Trainer'

        # To be populated in _initialize().
        self.step_op = None
        self.learn_rate_op = None
        self.grad_var_op = None
        self.train_op = None

        self._initialize()

    def _initialize(self):
        # TODO(bradyz): put somewhere better.
        bounds = [50000, 100000]
        values = [2e-3, 1e-3, 1e-4]

        with tf.variable_scope(self.scope):
            step_op = tf.Variable(0, name='step', trainable=False)
            learn_rate_op = tf.train.piecewise_constant(step_op, bounds, values)

            optimizer_op = tf.train.AdamOptimizer(learn_rate_op)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                grad_var_op = optimizer_op.compute_gradients(
                        self.network.loss_op,
                        var_list=self.network.weights)

                train_op = optimizer_op.apply_gradients(
                        grad_var_op, global_step=step_op)

        # Actually populated.
        self.step_op = step_op
        self.learn_rate_op = learn_rate_op
        self.grad_var_op = grad_var_op
        self.train_op = train_op

    def train(self, sess):
        sess.run(self.train_op, {self.network.is_training_op: True})
