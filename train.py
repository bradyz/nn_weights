import os

import numpy as np
import tensorflow as tf

import config
import dataprovider
import model



def get_inputs(provider_train, provider_valid, batch_size):
    with tf.name_scope('is_training'):
        is_training_op = tf.placeholder(tf.bool)

    with tf.name_scope('samples'):
        inputs_list = tf.cond(
                is_training_op,
                lambda: provider_train.get_inputs(batch_size),
                lambda: provider_valid.get_inputs(batch_size))

    return is_training_op, inputs_list


def train(experiment, provider_train, provider_valid):
    with tf.Session() as sess:
        experiment.ready_up(sess)

        # Multithreading.
        coord = tf.train.Coordinator()

        threads = list()
        threads += provider_train.create_threads(sess, coord)
        threads += provider_valid.create_threads(sess, coord)

        try:
            for _ in range(config.num_steps):
                if coord.should_stop():
                    break

                # Ugly please fix.
                step = sess.run(experiment.trainers[0].step_op)

                if step % config.checkpoint_steps == 0:
                    experiment.checkpoint(step)

                if step % config.save_steps == 0:
                    experiment.save(sess)

                experiment.train(sess)

            coord.request_stop()
        except Exception as e:
            print(e)
            coord.request_stop(e)
        finally:
            coord.join(threads)


def main():
    get_provider = lambda data_path, name, is_training: dataprovider.provider_factory(
            data_path, config.input_shape, is_training, name)

    provider_train = get_provider(config.train_path, 'train', True)
    provider_valid = get_provider(config.valid_path, 'valid', False)

    is_training_op, (images_op, labels_op) = get_inputs(
            provider_train, provider_valid, config.batch_size)

    network1 = model.VanillaNetwork(images_op, config.num_classes, is_training_op,
            256,
            save_path=os.path.join(config.log_dir, config.model_name + '_1'),
            labels_op=labels_op,
            scope='Vanilla1')

    network2 = model.VanillaNetwork(images_op, config.num_classes, is_training_op,
            32,
            save_path=os.path.join(config.log_dir, config.model_name + '_2'),
            labels_op=labels_op,
            scope='Vanilla2')

    experiment = model.Experiment(is_training_op, config.log_dir)
    experiment.add(network1)
    experiment.add(network2)

    train(experiment, provider_train, provider_valid)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    main()
