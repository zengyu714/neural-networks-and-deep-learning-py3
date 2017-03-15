"""network3.py
~~~~~~~~~~~~~~
A TensorFlow-based program for training and running simple neural networks.

Reference: mnist_with_summaries.py
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.

"""
import mnist_loader
import tensorflow as tf
import random
import numpy as np

# --------------------------------------------------------------------
# layer functions

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def fc_layer(inputs, input_nums, output_nums, activation_func=tf.nn.relu):
    W = weight_variable([input_nums, output_nums])
    b = bias_variable([output_nums])
    z = tf.matmul(inputs, W) + b
    if activation_func is not None:
        z = activation_func(z)
    return z

def main(_):

    # Import data
    training_data, validation_data, test_data = mnist_loader.load_data()

    # Construct convolutional network
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # conv1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # conv2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fc
    flatten = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc = fc_layer(flatten, 7*7*64, 1000, tf.nn.relu)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(h_fc, keep_prob)

    # output
    y_conv = fc_layer(dropped, 1000, 10, activation_func=None)

    # Compute loss
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Training
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    def next_batch(data, step, batch_size):
        size = len(data)
        if size < batch_size:
            raise ValueError("Batch size should be larger than dataset: {:<7d}".format(size))
        if step % 500 == 0:
            random.shuffle(data)
        start = step % size
        end = step + batch_size
        if end <= size:
            temp = data[start: end]
        else:
            temp = data[start - batch_size : start]
        # looks wierd because the input data is a list of tuples
        temp_x, temp_y = list(zip(*temp))
        return np.squeeze(np.array(temp_x)), np.squeeze(np.array(temp_y))


    for i in range(10000):
        batch_x, batch_y = next_batch(training_data, i, 100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            test_batch_x, test_batch_y = next_batch(training_data, i, 100)
            test_accuracy = accuracy.eval(feed_dict={x: test_batch_x, y_: test_batch_y, keep_prob: 1.0})
            print("step %d, training accuracy %7.4f, test accuracy %7.4f" % (i, train_accuracy, test_accuracy))

        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

if __name__ == '__main__':
    tf.app.run()
