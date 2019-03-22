"""
Original Code
https://github.com/pumpikano/tf-dann/create_mnistm.py
modified by Jaeyoon Yoo.
"""

from __future__ import absolute_import, division, print_function

import numpy as np


def compose_image(digit, background):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)

    bg = background[x:x + dw, y:y + dh]
    return np.abs(bg - digit).astype(np.uint8)


def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X, background_data):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    rand = np.random.RandomState(np.random.randint(100000))
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):

        if i % 1000 == 0:
            print('Processing example', i)

        bg_img = rand.choice(background_data)

        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d

    return X_
