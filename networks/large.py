"""
Original Code
https://github.com/RuiShu/dirt-t/codebase/models/nns/large.py
Modified by Changhwa Park.
"""

import tensorflow as tf
from tensorbayes.layers import dense, conv2d, avg_pool, max_pool, batch_norm, instance_norm
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.ops.nn_ops import dropout
from utils import leaky_relu, noise
from data.dataset import get_attr


class large(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        _, _, _, _, _, _, self.nc = get_attr(FLAGS.src, FLAGS.trg)

    def classifier(self, x, phase, enc_phase=True, trim=0, scope='class', reuse=tf.AUTO_REUSE, internal_update=False,
                   getter=None):
        with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
            with arg_scope([leaky_relu], a=0.1), \
                 arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=phase), \
                 arg_scope([batch_norm], internal_update=internal_update):
                preprocess = instance_norm if self.FLAGS.inorm else tf.identity
                layout = [
                    (preprocess, (), {}),
                    (conv2d, (96, 3, 1), {}),
                    (conv2d, (96, 3, 1), {}),
                    (conv2d, (96, 3, 1), {}),
                    (max_pool, (2, 2), {}),
                    (dropout, (), dict(training=phase)),
                    (noise, (1,), dict(phase=phase)),
                    (conv2d, (192, 3, 1), {}),
                    (conv2d, (192, 3, 1), {}),
                    (conv2d, (192, 3, 1), {}),
                    (max_pool, (2, 2), {}),
                    (dropout, (), dict(training=phase)),
                    (noise, (1,), dict(phase=phase)),
                    (conv2d, (192, 3, 1), {}),
                    (conv2d, (192, 3, 1), {}),
                    (conv2d, (192, 3, 1), {}),
                    (avg_pool, (), dict(global_pool=True)),
                    (dense, (self.nc,), dict(activation=None))
                ]

                if enc_phase:
                    start = 0
                    end = len(layout) - trim
                else:
                    start = len(layout) - trim
                    end = len(layout)

                for i in range(start, end):
                    with tf.variable_scope('l{:d}'.format(i)):
                        f, f_args, f_kwargs = layout[i]
                        x = f(x, *f_args, **f_kwargs)

        return x

    def feature_discriminator(self, x, phase, C=1, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('disc/feat', reuse=reuse):
            with arg_scope([dense], activation=tf.nn.relu):  # Switch to leaky?

                x = dense(x, 100)
                x = dense(x, C, activation=None)

        return x
