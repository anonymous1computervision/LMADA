# Local Manifold Adversarial Domain Adaptation

# Import required packages
import os

import tensorflow as tf

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

## Hyper-parameters
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('dp', 0.5, 'Dropout keep probability')
flags.DEFINE_integer('iter', 100000, 'Maximum iteration')
flags.DEFINE_integer('bs', 128, 'Batch size')

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

    # Define model name
    setup_list = [
        f"src={FLAGS.src}",
        f"trg={FLAGS.trg}",
        f"nn={FLAGS.nn}",
        f"ls={FLAGS.lr}"
    ]
    model_name = '_'.join(setup_list)
    print(model_name)


if __name__ == '__main__':
    tf.app.run()
