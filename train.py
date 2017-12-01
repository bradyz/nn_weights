import argparse

import numpy as np
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset

import model
import ops
import settings


def get_data(is_train, batch_size, num_samples=None):
    data = dataset.Cifar10('train' if is_train else 'test')
    mean = data.get_per_pixel_mean()

    if num_samples is not None:
        data = DataFromList(data.data[:num_samples], shuffle=False)

    augmentors = [imgaug.MapImage(lambda x: x - mean)]

    if is_train:
        augmentors += [
                # imgaug.CenterPaste((40, 40)),
                # imgaug.RandomCrop((32, 32)),
                imgaug.Flip(horiz=True),
                # imgaug.Brightness(10, clip=True),
                # imgaug.GaussianNoise(),
                # imgaug.Contrast((0.9, 1.1), clip=True),
                # imgaug.Saturation(0.2),
                # imgaug.SaltPepperNoise(),
            ]

    data = AugmentImageComponent(data, augmentors)

    if is_train:
        data = RepeatedDataPoint(data, 3)

    data = BatchData(data, batch_size, remainder=not is_train)

    if is_train:
        data = PrefetchData(data, 3, 2)

    return data


def get_config(restore_path=None, batch_size=128):
    dataset_train = get_data(True, batch_size)
    dataset_valid = get_data(False, batch_size)

    network = model.Network()

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ops.ValidationCallback(dataset_valid),
            ops.InitializationCallback(dataset_valid, network),
            # ModelSaver(3),
            ScheduledHyperParamSetter(
                'learning_rate', [
                    (1, 1e-3),
                    (50, 1e-4),
                    (100, 1e-5),
                ])],
        model=network,
        steps_per_epoch=1000,
        max_epoch=150,
        session_init=TryResumeTraining() if not restore_path else SaverRestore(restore_path)
    )


def get_init_config(batch_size=32):
    dataset_train = get_data(True, batch_size, 16)
    dataset_valid = get_data(False, batch_size, 16)
    network = model.Pretrain()

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ops.InitializationCallback(dataset_valid, network),
            ModelSaver(1),
            ScheduledHyperParamSetter(
                'learning_rate', [
                    (1, 1e-4)
                ])],
        model=network,
        steps_per_epoch=1000,
        max_epoch=1000,
        session_init=TryResumeTraining()
    )


def main(initialize, log_dir, restore_path):
    logger.set_logger_dir(log_dir)

    config = get_init_config() if initialize else get_config(restore_path)
    trainer = QueueInputTrainer()

    launch_train_with_config(config, trainer)


def train_all():
    for x, nonlinearity in settings.nonlinearities.items():
        for y, initializer in settings.initializers.items():
            for z, optimizer in settings.optimizers.items():
                settings.nonlinearity = nonlinearity
                settings.initializer = initializer
                settings.optimizer = optimizer

                with tf.Graph().as_default():
                    main(False, 'log_init/%s_%s_%s' % (x, y, z), None)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--initialize', dest='initialize', action='store_true')
    # parser.add_argument('--log')
    # parser.add_argument('--restore_path', default=None)
    # parser.set_defaults(initialize=False)
    #
    # args = parser.parse_args()
    #
    # initialize = args.initialize
    # log_dir = args.log
    # restore_path = args.restore_path
    #
    # main(initialize, log_dir, restore_path)

    train_all()
