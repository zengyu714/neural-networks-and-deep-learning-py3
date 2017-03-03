"""
network.py
~~~~~~~~~~
IT WORKS
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  I have optimized it a little bit.

"""

import random
import numpy as np

class Network(object):

    def __init__(self, shape):
        """The sequence `shape contains the number of neurons in
        each layer, e.g. [2, 3, 1] would be a three-layer network,
        with the first layer containing 2 neurons, the second layer
        3 neurons, and the third layer 1 neuron.
        The biases and weights for the network are initialized
        randomly, using a Gaussian distribution with mean 0, and variance 1.

        """

        self.num_layers = len(shape)
        self.shape = shape        # input layer has no bias ↓↓
        self.biases = [np.random.randn(y, 1) for y in shape[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]

    def forward(self, x):
        """Return the output of the network if `x` is input."""

        for b, w in zip(self.biases, self.weights):
            # The matmul function implements the semantics of the @ operator
            # introduced in Python 3.5
            x = sigmoid(w @ x + b)
        return x

    def SGD(self, training_data, epochs, batch_size, lr_rate, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.
        training_data:  a list of tuples (raw_input, label)
        lr_rate      :  learning rate
        test_data    :  if is provided, the network will be evaluated after each
                        epoch, and partial progress printed out. This is useful for
                        tracking progress, but slows things down substantially.

        """

        if test_data:
            num_test = len(test_data)
        num_train = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k: k+batch_size] for k in range(0, num_train, batch_size)]
            for batch in batches:
                self.update_batch(batch, lr_rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j + 1, self.evaluate(test_data), num_test))
            else:
                print("Epoch {0} complete".format(j + 1))

    def update_batch(self, batch, lr_rate):
        """Update the network's weights and biases using gradient descent using
        backpropagation to a single mini batch.

        batch: a list of tuples (raw_input, label)

        """

        batch_derivative_w = [np.zeros(w.shape) for w in self.weights]
        batch_derivative_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            single_derivative_w, single_derivative_b = self.backprop(x, y)
            batch_derivative_w = [s_dw + b_dw for s_dw, b_dw in
                                    zip(single_derivative_w,batch_derivative_w)]
            batch_derivative_b = [s_db + b_db for s_db, b_db in
                                    zip(single_derivative_b,batch_derivative_b)]

        self.weights = [w - lr_rate/len(batch) * dw
                             for w, dw in zip(self.weights, batch_derivative_w)]
        self.biases = [b - lr_rate/len(batch) * db
                             for b, db in zip(self.biases, batch_derivative_b)]

    def backprop(self, x, y):
        """Return a tuple
        (dw, db) :
        + gradient for the cost function C_x.
        + layer-by-layer lists of numpy arrays,
          similar to `self.weights` and `self.biases`.

        """

        # initialize
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]

        a   =  x
        a_s = [a]   # list to store all the activations, layer by layer
        z_s = [ ]   # list to store all the z vectors  , layer by layer

         # feedforward
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            z_s.append(z)
            a = sigmoid(z)
            a_s.append(a)

        # backward
        err = self.derivative_C(a_s[-1], y) * derivative_sigmoid(z_s[-1])
        dw[-1] = err @ a_s[-2].T  # transpose
        db[-1] = err

        # Note that the variable l
        # l = 1 means the last layer of neurons,
        # l = 2 is the second-last layer, and so on.
        for layer in range(2, self.num_layers):
            z = z_s[-layer]
            d_sig = derivative_sigmoid(z)
            err = self.weights[-layer + 1].T @ err * d_sig
            dw[-layer] = err @ a_s[-layer - 1].T
            db[-layer] = err
        return (dw, db)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.

        """

        test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for x, y in test_results)

    def derivative_C(self, output, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations, when Cost is a quadratic function.

        
        """

        return output - y

#------------------------------------------------------------------------------------

# helper function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z)*(1 - sigmoid(z))
