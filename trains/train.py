from data.dataset import get_data
import matplotlib.pyplot as plt
import numpy as np


def train(M, FLAGS):
    print("train called")
    data = get_data(FLAGS.src, FLAGS)
    print(f"The number of train data is         {data.train_num}")
    print(f"The number of test data is          {data.test_num}")
    print(f"The number of validation data is    {data.val_num}")
    images, labels = data.train.next_batch(128)
    print(f"The shape of the batch images is    {images.shape}")
    print(f"The shape of the batch labels is    {labels.shape}")
    print(np.max(images))
    print(np.min(images))

    # This code shows data shape and plots it.
    for i in range(1):
        first_data = images[i]
        print(first_data.shape)
        trans_data = np.transpose(first_data, (1, 2, 0)).astype(np.float32)
        #trans_data = np.tile(np.transpose(first_data, (1, 2, 0)).astype(np.float32), (1, 1, 3))
        print(trans_data.shape)
        plt.imshow(trans_data)
        plt.show()
        print(f"The label of the image is           {labels[i]}")
        print(f"The label of the image is           {np.argmax(labels[i])}")
