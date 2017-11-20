import argparse

import numpy as np
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset

import model
import ops


def get_data(is_train, batch_size):
    data = dataset.Cifar10('train' if is_train else 'test')
    mean = data.get_per_pixel_mean()

    augmentors = [imgaug.MapImage(lambda x: x - mean)]

    if is_train:
        augmentors += [
                imgaug.CenterPaste((40, 40)),
                imgaug.RandomCrop((32, 32)),
                imgaug.Flip(horiz=True)]

    data = AugmentImageComponent(data, augmentors)
    data = BatchData(data, batch_size, remainder=not is_train)

    if is_train:
        data = PrefetchData(data, 3, 2)

    return data


def get_config(batch_size=256):
    dataset_train = get_data(True, batch_size)
    dataset_valid = get_data(False, batch_size)

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            # ops.VisualizeTestSet(dataset_valid),
            ModelSaver(3),
            ScheduledHyperParamSetter(
                'learning_rate', [
                    (1, 2e-3),
                    (100, 1e-3),
                    (250, 2e-4),
                    (500, 1e-4),
                ])],
        model=model.Network('Vanilla'),
        steps_per_epoch=1000,
        max_epoch=1000,
        session_init=TryResumeTraining()
    )


def main(log_dir):
    logger.set_logger_dir(log_dir)

    config = get_config()
    trainer = QueueInputTrainer()

    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log')

    args = parser.parse_args()

    log_dir = args.log

    main(log_dir)
