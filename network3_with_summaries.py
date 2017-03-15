"""network3.py
~~~~~~~~~~~~~~
A TensorFlow-based program for training and running simple neural networks.
Add with summaries displayed in Tensorboard.

Reference: mnist_with_summaries.py
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.

"""
import tensorflow as tf

import random
import pickle
import gzip
import numpy as np

log_dir = '/tmp/mnist/logs/'

# --------------------------------------------------------------------
# Load data

def one_hot(true_label):
    one_hot_list = [int(i == true_label) for i in range(10)]
    return np.reshape(one_hot_list, (10, 1))

def load_data():
    with gzip.open("mnist.pkl.gz", "rb") as f:
        total_data = pickle.load(f, encoding="latin1")
    for data in total_data:
        data_x = [x.reshape(784, 1) for x in data[0]]
        data_y = [one_hot(y) for y in data[1]]
        data = list(zip(data_x, data_y))
        yield data


def train():
    # Import data
    training_data, validation_data, test_data = load_data()

    sess = tf.InteractiveSession()
    # Create a multilayer model
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor, for TensorBoard visualization.
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def fc_layer(inputs, input_nums, output_nums, layer_name, activation_func=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                W = weight_variable([input_nums, output_nums])
                variable_summaries(W)
            with tf.name_scope('biases'):
                b = bias_variable([output_nums])
                variable_summaries(b)
            with tf.name_scope('pre_activation'):
                z = tf.matmul(inputs, W) + b
                tf.summary.histogram('z', z)
            if activation_func is not None:
                z = activation_func(z)
                tf.summary.histogram('activation', z)
            return z

    with tf.name_scope('conv1'):
        W = weight_variable([5, 5, 1, 32])
        b = bias_variable([32])
        relu = tf.nn.relu(conv2d(image_shaped_input, W) + b)
        # relu[..., [1]]
        tf.summary.image('relu', relu[..., 1, None], 10)
        pool = max_pool_2x2(relu)

    with tf.name_scope('conv2'):
        W = weight_variable([5, 5, 32, 64])
        b = bias_variable([64])
        relu = tf.nn.relu(conv2d(pool, W) + b)
        # relu[..., [1]]
        tf.summary.image('relu', relu[..., 1, None], 10)
        pool = max_pool_2x2(relu)

    with tf.name_scope('flatten_reshape'):
        flatten = tf.reshape(pool, [-1, 7*7*64])

    fc1 = fc_layer(flatten, 7*7*64, 1000, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(fc1, keep_prob)

    y_conv = fc_layer(dropped, 1000, 10, 'output', activation_func=None)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            print(tf.shape(y_conv),'-'*30)
            print(tf.shape(y_),'-'*30)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + 'test')
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
        if i % 10 == 0: # Record summaries and test-set accuracy
            test_batch_x, test_batch_y = next_batch(test_data, i, 100)
            summary, acc = sess.run([merged, accuracy],
                    feed_dict={x: test_batch_x, y_: test_batch_y, keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:           # Record execution stats
            batch_x, batch_y = next_batch(training_data, i, 100)
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                              feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5},
                              options=run_options,
                              run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
            else:       # Record a summary
                summary, _ = sess.run([merged, train_step],
                              feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    train()

if __name__ == '__main__':
    tf.app.run(main=main)
