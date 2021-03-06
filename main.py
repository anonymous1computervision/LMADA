# Local Manifold Adversarial Domain Adaptation

import os

import tensorflow as tf

from models.model import lmada
from trains.train import train

PATH = '/home/omega/mycodes/LMADA/results'

# Define flag arguments
flags = tf.app.flags

## Data
flags.DEFINE_string('src', 'svhn', 'Source domain name')
flags.DEFINE_string('trg', 'mnist', 'Target domain name')

## Architecture
flags.DEFINE_string('nn', 'small', 'Network architecture')
flags.DEFINE_string('rootdir', PATH, 'Home directory for results')
flags.DEFINE_string('logdir', 'log', 'Log directory')
flags.DEFINE_string('datadir', '/home/omega/datasets', 'Directory for datasets')
flags.DEFINE_boolean('val', True, 'Include validation set or not')

## Hyper-parameters
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('dp', 0.5, 'Dropout keep probability')
flags.DEFINE_integer('iter', 100000, 'Maximum iteration')
flags.DEFINE_integer('bs', 128, 'Batch size')
flags.DEFINE_boolean('zc', False, 'Zero centering of data')
flags.DEFINE_boolean('inorm', False, 'Instance normalization')

## Load previously trained session
flags.DEFINE_boolean('load', False, 'Determines whether to load previous checkpoint or not')
flags.DEFINE_string('ckptdir', '', 'Previous checkpoint directory')

## Others
flags.DEFINE_string('gpu', '0', 'GPU number')

FLAGS = flags.FLAGS


def main(_):
    # Define GPU configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    # Define model name
    setup_list = [
        f"src={FLAGS.src}",
        f"trg={FLAGS.trg}",
        f"nn={FLAGS.nn}",
        f"lr={FLAGS.lr}",
        f"zc={FLAGS.zc}"
    ]
    model_name = '___'.join(setup_list)
    print(model_name)

    # TODO: Should consider tensorboard using 'train_writer'
    """
    if FLAGS.log:
        tf.logging.set_verbosity('INFO')
    """

    # TODO: Loading pretrained model or saved model

    M = lmada(FLAGS, gpu_config)
    M.sess.run(tf.global_variables_initializer())

    train(M, FLAGS, model_name)


if __name__ == '__main__':
    tf.app.run()
