# standard library
import pickle
import gzip

import numpy as np

def load_data():
    """
    training_data, validation_data, test_data = load_data()

    This function generates the MNIST data sequentially
    e.g.
    training_data:   (raw_pixels, labels)
                      both are **np.ndarray** type with size 50000 x 784 and 50000 x 10
    validation_data: (raw_pixels, labels)
    /test_data        both are **np.ndarray** type with size 10000 x 784 and 10000 x 1

    """
    with gzip.open("mnist.pkl.gz", "rb") as f:
        total_data = pickle.load(f, encoding="latin1")
        # total_data contains training_data, validation_data, test_data
        # e.g.
        # training_data[0].shape == (50000, 784)
        # training_data[1].shape == (50000, 1)

    for data in total_data:
        data_x = data[0]
        data_y = data[1]

        # if is training_data, make its label one-hot
        # --------------------------------------------------------------
        # def one_hot(true_label):
        #     return np.array([int(i == true_label) for i in range(10)])
        # --------------------------------------------------------------

        if len(data[0]) == 50000:
            data_y = [[int(i == y) for i in range(10)]
                          for y in data[1]]

        data = (data_x, data_y)
        yield data
