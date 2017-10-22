import tensorflow as tf
import numpy as np


def residual_block(x, is_training_op, scope):
    k = x.shape.as_list()[-1]

    with tf.variable_scope(scope):
        x_id = x

        x = batch_norm(x, is_training_op)
        x = relu(x)
        x = conv3x3(x, k, post='_1')

        x = batch_norm(x, is_training_op)
        x = relu(x)
        x = conv3x3(x, k, post='_2')

        x = x_id + x

    return x


def down_block(x, k, is_training_op, scope):
    with tf.variable_scope(scope):
        x = conv3x3(x, k, strides=2, post='2')
        x = batch_norm(x, is_training_op)
        x = relu(x)
        x = dropout(x, is_training_op)

    return x


def dense_conv(x, k, is_training_op, scope, activation=True):
    with tf.variable_scope(scope):
        x = conv1x1(x, k)
        x = batch_norm(x, is_training_op)

        if activation:
            x = relu(x)
            x = dropout(x, is_training_op, 0.8)

    return x


def dense_block(x, k, is_training_op, scope, activation=True, drop=0.5):
    with tf.variable_scope(scope):
        x = dense(x, k)

        if activation:
            # x = batch_norm(x, is_training_op)
            x = relu(x)
            x = dropout(x, is_training_op, drop)

    return x


def flatten(x, post=''):
    if len(x.shape.as_list()) == 2:
        return x

    b, h, w, c = x.shape.as_list()

    with tf.variable_scope('flatten/%s' % post):
        x = tf.reshape(x, [b, h * w * c])

    return x


def global_average_pooling(x):
    with tf.name_scope('global_average_pooling'):
        y = tf.reduce_mean(x, [1, 2])

    return y


def global_max_pooling(x):
    with tf.name_scope('global_max_pooling'):
        y = tf.reduce_max(x, [1, 2])

    return y


def max_pool(x):
    with tf.name_scope('max_pool'):
        y = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    return y


def conv3x3(x, c_out, strides=1, post=''):
    return conv(x, 3, c_out, strides, post='_3x3' + post)


def conv1x1(x, c_out, strides=1, post=''):
    return conv(x, 1, c_out, strides, post='_1x1' + post)


def relu(x):
    with tf.name_scope('relu'):
        y = tf.nn.relu(x)

    return y


def conv(x, k, c_out, strides=1, post=''):
    with tf.variable_scope('conv%s' % post):
        W = tf.get_variable('W', [k, k, x.shape[-1], c_out],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(1.0))
        b = tf.get_variable('b', [c_out],
                initializer=tf.constant_initializer(0.0),
                regularizer=tf.contrib.layers.l2_regularizer(1.0))

        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = x + b

    return x


def dense(x, output_size, post=''):
    with tf.variable_scope('dense%s' % post):
        W = tf.get_variable('W', [x.shape[1], output_size], tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.0))
        b = tf.get_variable('b', [output_size], tf.float32,
                            initializer=tf.constant_initializer(0.0),
                            regularizer=tf.contrib.layers.l2_regularizer(1.0))

        x = tf.matmul(x, W)
        x = x + b

    return x


def batch_norm(x, is_training_op):
    return tf.layers.batch_normalization(x, training=is_training_op)


def dropout(x, is_training_op, drop_prob=0.5):
    return tf.layers.dropout(x, drop_prob, training=is_training_op)


def batch_data_augment(x_op):
    """
    Expects tensor of shape (batch_size, height, width, 3).
    """
    return tf.map_fn(data_augment, x_op)


def data_augment(x_op):
    x_op = tf.image.random_flip_left_right(x_op)
    x_op = tf.image.random_brightness(x_op, max_delta=2.0/255.0)
    x_op = tf.image.random_saturation(x_op, lower=0.5, upper=1.5)
    x_op = tf.image.random_hue(x_op, max_delta=0.2)
    x_op = tf.image.random_contrast(x_op, lower=0.5, upper=1.5)

    return x_op


def accuracy(pred_op, true_op):
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(
                tf.to_float(tf.equal(tf.round(pred_op), true_op)))

    return accuracy


def confusion_image(pred_op, true_op, num_classes):
    with tf.name_scope('confusion_matrix'):
        confusion_op = tf.confusion_matrix(
                true_op, pred_op, num_classes)
        confusion_op = tf.expand_dims(confusion_op, axis=0)
        confusion_op = tf.expand_dims(confusion_op, axis=-1)
        confusion_op = tf.to_float(confusion_op)

    return confusion_op


def make_Cx(c):
    Cx = tf.zeros()
    return np.array([[  0.0, -c[2],  c[1]],
                     [ c[2],   0.0, -c[0]],
                     [-c[1],  c[0],  0.0]])


def make_R_unstable(c, iterations=50):
    A = make_Cx(c)

    n = A.shape[0]

    result = np.zeros((n, n))
    current = np.eye(n)
    current_bot = 1.0

    for i in range(1, iterations+1):
        result += (1.0 / current_bot) * current

        current = np.dot(current, A)
        current_bot *= i

    return result


def make_R(u, eps=1e-12):
    n = int(u.shape[0])

    u = u / tf.norm(u + eps)
    u = tf.expand_dims(u, 1)

    M = tf.eye(n) - 2.0 * tf.matmul(u, tf.transpose(u))
    # M = tf.concat([M[:,:-1], -M[:,-1:]], axis=1)

    return M


def make_S(v, input_size, output_size):
    n = int(v.shape[0])

    S = tf.diag(tf.square(v))
    S = tf.pad(S, [(0, input_size - n), (0, output_size - n)])
    S = S[:input_size,:output_size]

    return S


def my_dense(x, output_size, post=''):
    input_size = int(x.shape[1])

    with tf.variable_scope('dense%s' % post):
        u = tf.get_variable('u', [input_size], tf.float32,
                initializer=tf.random_normal_initializer(stddev=1e-3))
        v = tf.get_variable('v', [min(input_size, output_size)], tf.float32,
                initializer=tf.constant_initializer(1.0))
        b = tf.get_variable('b', [output_size],
                initializer=tf.constant_initializer(0.0),
                regularizer=tf.contrib.layers.l2_regularizer(1.0))

        R = make_R(u)
        S = make_S(v, input_size, output_size)
        W = tf.matmul(R, S)

        x = tf.matmul(x, W) + b

    return x


def my_dense_block(x, k, is_training_op, scope, activation=True, drop=0.5):
    with tf.variable_scope(scope):
        x = my_dense(x, k)

        if activation:
            # x = batch_norm(x, is_training_op)
            x = relu(x)
            x = dropout(x, is_training_op, drop)

    return x
