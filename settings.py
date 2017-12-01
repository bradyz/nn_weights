import tensorflow as tf

from tensorpack import *


nonlinearity = None
normalize = None
optimizer = None
initializer = None

nonlinearities = {
        'no_bn': tf.nn.relu,
        'bn': lambda l: BNReLU('bn_relu', l)
        }

initializers = {
        'std1': lambda: tf.random_normal_initializer(stddev=1e-2),
        'std2': lambda: tf.random_normal_initializer(stddev=1e-3),
        'xavier': tf.contrib.layers.xavier_initializer,
        'he': tf.contrib.layers.variance_scaling_initializer
        }

optimizers = {
        'sgd': lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
        'adam': tf.train.AdamOptimizer
        }
