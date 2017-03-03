# coding: utf-8
"""network2.py
~~~~~~~~~~~~~~
An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.

Improvements include
1. cross-entropy cost function
2. regularization
3. better initialization of network weights

"""

import random
import json
import sys

import numpy as np

# --------------------------------------------------------------------
# @staticmethod

# neither `self` (the object instance) nor `cls` (the class)
# is implicitly passed as the first argument,
# so they can be called from an instance **or the class**
# --------------------------------------------------------------------

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output `a` and desired output `y`.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a ,y):
        """Return the error delta from the output layer."""
        return (a-y) * derivative_sigmoid(z)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        # replace nan with zero and inf with finite numbers
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        # `z` to make the interface consistent with the delta method for other
        # cost classes
        return a - y

class Network(object):

    def __init__(self, shape, cost=CrossEntropyCost):
        """The sequence `shape` contains the number of neurons in
        each layer, e.g. [2, 3, 1] would be a three-layer network,
        with the first layer containing 2 neurons, the second layer
        3 neurons, and the third layer 1 neuron.
        The biases and weights for the network are initialized
        randomly, using `self.default_weight_initializer()`

        """

        self.num_layers = len(shape)
        self.shape = shape
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """Initialize with a Gaussian distribution
        each weight: miu = 0, sigma = 1/sqrt(n_in)
        each bias  : miu = 0, sigma = 1

        """
        self.biases = [np.random.randn(y, 1) for y in self.shape[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.shape[:-1], self.shape[1:]) ]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.shape[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.shape[:-1], self.shape[1:])]

    def forward(self, a):
        """Return the output of the network if `a` is input."""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def SGD(self, training_data, epochs, batch_size, lr_rate,
           reg_lambda = 0,
           evaluation_data=None,
           monitor_evaluation_cost=False,
           monitor_evaluation_accuracy=False,
           monitor_training_cost=False,
           monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient descent.
        training_data   :  a list of tuples (raw_input, label)
        lr_rate         :  learning rate
        reg_lambda      :  control the regularization of weights
        evaluation_data :  accpects either the validation or test data
        monitor_xx_xx   :  monitor the cost and accuracy on either the
                          evaluation data or training data, by setting FLAG

        This method returns a tuple containing four lists, per-epoch
        ([costs on the evaluation data],
         [accuracies on the evaluation data],
         [costs on the training data],
         [accuracies on the training data])

        Note that the lists are empty if the corresponding flag is not set.

        """
        if evaluation_data:
            num_eval = len(evaluation_data)
        num_train = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k: k+batch_size] for k in range(0, num_train, batch_size)]
            for batch in batches:
                self.update_batch(batch, lr_rate, reg_lambda, len(training_data))
            print("Epoch {0} complete".format(j + 1))
            if monitor_training_cost:
                cost = self.total_cost(training_data, reg_lambda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, num_train))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, reg_lambda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evalution data: {} / {}".format(accuracy, num_eval))

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy


    def update_batch(self, batch, lr_rate, reg_lambda, n):
        """Update the network's weights and biases using gradient descent using
        backpropagation to a single mini batch.

        batch : a list of tuples (raw_input, label)
        n     : the total size of the training data set,
                used in weights update with regularization

        """
        batch_derivative_w = [np.zeros(w.shape) for w in self.weights]
        batch_derivative_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            single_derivative_w, single_derivative_b = self.backprop(x, y)
            batch_derivative_w = [s_dw + b_dw for s_dw, b_dw in
                                    zip(single_derivative_w,batch_derivative_w)]
            batch_derivative_b = [s_db + b_db for s_db, b_db in
                                    zip(single_derivative_b,batch_derivative_b)]

        self.weights = [(1-reg_lambda*lr_rate/n)*w - lr_rate/len(batch)*dw
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
        err = (self.cost).delta(z_s[-1], a_s[-1], y)
        dw[-1] = err @ a_s[-2].T
        db[-1] = err

        for layer in range(2, self.num_layers):
            z = z_s[-layer]
            d_sig = derivative_sigmoid(z)
            err = self.weights[-layer + 1].T @ err * d_sig
            dw[-layer] = err @ a_s[-layer - 1].T
            db[-layer] = err

        return (dw, db)

    def accuracy(self, data, convert=False):
        """
        convert: the single label representation --> one_hot representation

        """
        if convert:
            res = [(np.argmax(self.forward(x)), np.argmax(y))
                  for (x, y) in data]
        else:
            res = [(np.argmax(self.forward(x)), y)
                  for (x, y) in data]
        return sum(int(x == y) for (x, y) in res)

    def total_cost(self, data, reg_lambda, convert=False):

        cost = 0
        for x, y in data:
            a = self.forward(x)
            if convert:
                y = one_hot(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(reg_lambda/len(data))*sum(
                    np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save_net(self, filename):
        data = {"shape"  : self.shape,
                "weights": [w.tolist() for w in self.weights],
                "biases" : [b.tolist() for b in self.biases],
                "cost"   : str(self.cost.__name__)}
        with open(filename, 'w+') as f:
            json.dump(data, f)

#----------------------------------------------------------------------------

# helper function
def load_net(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["shape"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def one_hot(true_label):
    one_hot_list = [int(i == true_label) for i in range(10)]
    return np.reshape(one_hot_list, (10, 1))

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z)*(1 - sigmoid(z))

# --------------------------------------------------------------------
# test

# import mnist_loader
# from network2 import Network, CrossEntropyCost
#
# training_data, validation_data, test_data = mnist_loader.load_data()
#
# net = Network([784, 30, 10], cost=CrossEntropyCost)
# net.large_weight_initializer()
#
# net.SGD(training_data, 30, 10, 0.1, reg_lambda= 5.0,
#         evaluation_data=validation_data,
#         monitor_evaluation_accuracy=True, monitor_evaluation_cost=True,
#         monitor_training_accuracy=True, monitor_training_cost=True)
