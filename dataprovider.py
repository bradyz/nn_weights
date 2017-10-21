import pickle

import tensorflow as tf
import numpy as np


def _get_pairs(data_paths):
    images = list()
    labels = list()

    for data_path in data_paths:
        with open(data_path, 'rb') as fd:
            data = pickle.load(fd, encoding='bytes')

            # np.array of shape (10000, 3072).
            images_subset = data[b'data']
            images_subset = images_subset.reshape((-1, 3, 32, 32))
            images_subset = images_subset.transpose((0, 2, 3, 1))

            # list of length 10000.
            labels_subset = data[b'labels']

            # Add to full collection
            images.extend(images_subset)
            labels.extend(labels_subset)

    images = np.float32(images)
    labels = np.int32(labels)

    return [images, labels]


def provider_factory(data_paths, input_shape, is_training, name):
    datagen = CifarDatagen(data_paths, input_shape, is_training)
    provider = AsyncProvider(datagen, name)

    return provider


class CifarDatagen(object):
    def __init__(self, data_paths, input_shape, is_training):
        self.image_label = _get_pairs(data_paths)
        self.is_training = is_training

        self.index = 0

        self._dtypes = [tf.float32, tf.int32]
        self._shapes = [input_shape, []]

    def __next__(self):
        image = self.image_label[0][self.index]
        label = self.image_label[1][self.index]

        self.index = (self.index + 1) % len(self.image_label[0])

        return image, label

    def __iter__(self):
        return self

    def get_dtypes(self):
        return self._dtypes

    def get_shapes(self):
        return self._shapes

    def augment(self, data_op):
        if self.is_training:
            return data_op

        return data_op


class AsyncProvider(object):
    def __init__(self, generator, scope, cap=2048, min_after=512):
        dtypes = generator.get_dtypes()
        shapes = generator.get_shapes()

        with tf.name_scope(scope):
            self.queue = tf.RandomShuffleQueue(cap, min_after, dtypes, shapes)

            data_op = tf.py_func(lambda: next(generator), [], dtypes)
            data_op = generator.augment(data_op)

            for i, x_op in enumerate(data_op):
                x_op.set_shape(shapes[i])

            enqueue_op = self.queue.enqueue(data_op)

            self.runner = tf.train.QueueRunner(self.queue, [enqueue_op])

            tf.summary.scalar('capacity', self.queue.size())

    def get_inputs(self, batch_size):
        return self.queue.dequeue_many(batch_size)

    def create_threads(self, sess, coord):
        return self.runner.create_threads(sess, coord, daemon=True, start=True)
