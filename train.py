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


def train(network, trainer, monitor, provider_train, provider_valid):
    with tf.Session() as sess:
        network.ready_up(sess)
        monitor.ready_up(sess)

        # Multithreading.
        coord = tf.train.Coordinator()

        threads = list()
        threads += provider_train.create_threads(sess, coord)
        threads += provider_valid.create_threads(sess, coord)

        try:
            for _ in range(config.num_steps):
                if coord.should_stop():
                    break

                step = sess.run(step_op)

                if step % config.checkpoint_steps == 0:
                    monitor.checkpoint(step)

                if step % config.save_steps == 0:
                    network.save()

                trainer.train(sess)

            coord.request_stop()
        except Exception as e:
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

    network = model.VanillaNetwork(images_op, config.num_classes, is_training_op,
            save_path=os.path.join(config.log_dir, config.model_name),
            labels_op=labels_op)

    print('Network weights.')
    print('\n'.join(sorted(map(lambda x: x.name, network.weights))))
    print()

    print('Weights to be saved.')
    print('\n'.join(sorted(map(lambda x: x.name, saved_vars))))

    train(provider_train, provider_valid,
               step_op, train_op, summary_op, is_training_op,
               saved_vars, save_path, config.Yearbook.log_dir,
               network, network.restore)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    main()
