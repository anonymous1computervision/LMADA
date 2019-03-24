import tensorflow as tf


class lmada(object):
    def __init__(self, FLAGS, gpu_config):
        print("============================LMADA model initialization started.============================")
        self.sess = tf.Session(config=gpu_config)
