import tensorflow as tf
import numpy as np

import ops


def make_R_householder(u, eps=1e-12):
    n = int(u.shape[0])

    u = u / tf.norm(u + eps)
    u = tf.expand_dims(u, 1)

    M = tf.eye(n) - 2.0 * tf.matmul(u, tf.transpose(u))
    M = tf.concat([M[:,:-1], -M[:,-1:]], axis=1)

    return M


def make_R_householder(A, eps=1e-12):
    """u is a skew symmetric matrix"""

    return M


def make_S(v, input_size, output_size):
    n = int(v.shape[0])

    S = tf.diag(v)
    S = tf.pad(S, [(0, input_size - n), (0, output_size - n)])
    S = S[:input_size,:output_size]

    return S


def my_dense(x, output_size, post=''):
    input_size = int(x.shape[1])

    with tf.variable_scope('dense%s' % post):
        u = tf.get_variable('u', [input_size], tf.float32,
                initializer=tf.random_normal_initializer(stddev=1e-3))
        v = tf.get_variable('v', [min(input_size, output_size)], tf.float32,
                initializer=tf.random_normal_initializer(stddev=1e-3))
        b = tf.get_variable('b', [output_size],
                initializer=tf.constant_initializer(0.0),
                regularizer=tf.contrib.layers.l2_regularizer(1.0))

        R = make_R_householder(u)
        S = make_S(v, input_size, output_size)
        W = tf.matmul(R, S)

        x = tf.matmul(x, W) + b

    return x


def constrained_dense(x, output_size, post=''):
    input_size = int(x.shape[1])

    with tf.variable_scope('dense%s' % post):
        w = tf.get_variable('w', [input_size, output_size], tf.float32,
                initializer=tf.random_normal_initializer(stddev=1e-3))

        R = make_R_householder(u)
        S = make_S(v, input_size, output_size)
        W = tf.matmul(R, S)

        x = tf.matmul(x, W) + b

    return x


x = tf.random_uniform([16, 10], -10.0, 10.0, dtype=tf.float32, seed=0)
x = my_dense(x, 256, 'block1')
x = my_dense(x, 256, 'block2')
x = my_dense(x, 256, 'block3')
x = my_dense(x, 10, 'block4')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

import pdb; pdb.set_trace()
