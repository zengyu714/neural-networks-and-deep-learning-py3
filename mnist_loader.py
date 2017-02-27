# standard library
import pickle
import gzip

import numpy as np

def one_hot(true_label):
    one_hot_list = [int(i == true_label) for i in range(10)]
    return np.reshape(one_hot_list, (10, 1))

def load_data():
    """
    training_data, validation_data, test_data = load_data()

    This function generates the MNIST data sequentially
    e.g.
    ------------------------------------------------------------------------
    training_data:   (raw_pixels, labels)
    type: numpy.ndarray with size (784, 1), numpy.ndarray with size (10, 1)
    ------------------------------------------------------------------------
    validation_data: (raw_pixels, label )
    test_data      : (raw_pixels, label )
    type: numpy.ndarray with size (784, 1), numpy.int64
    ------------------------------------------------------------------------
    """

    with gzip.open("mnist.pkl.gz", "rb") as f:
        total_data = pickle.load(f, encoding="latin1")
        # total_data contains training_data, validation_data, test_data
        # e.g.
        # training_data[0].shape == (50000, 784)
        # training_data[1].shape == (50000, 1)

    for data in total_data:
        data_x = [x.reshape(784, 1) for x in data[0]]
        data_y = data[1]

        # if is training_data, make its label one-hot
        if len(data_x) == 50000:
            data_y = [one_hot(y) for y in data[1]]

        data = list(zip(data_x, data_y))
        yield data
