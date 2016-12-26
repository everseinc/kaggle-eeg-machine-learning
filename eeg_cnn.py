import numpy
import random
import csv
import tensorflow as tf
from helper import *
from csv_manager import *

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 200
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
CATEGORY_NUM = 6
LEARNING_RATE = 0.0001
FILTER_SIZE = 5
FILTER_NUM = 32
FILTER_NUM2 = 64
FEATURE_DIM = 1024
KEEP_PROB = 0.5
TRAINING_LOOP = 260000
BATCH_SIZE = 100
SUMMARY_DIR = 'log_cnn_sl'
SUMMARY_INTERVAL = 100

with tf.Graph().as_default():
    with tf.name_scope('input'):
        y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM], name='labels')
        x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH], name='input_images')

    with tf.name_scope('convolution'):
        W_conv = weight_variable([FILTER_SIZE, FILTER_SIZE, 1, FILTER_NUM], name='weight_conv')
        b_conv = bias_variable([FILTER_NUM], name='bias_conv')
        x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
        h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)

    with tf.name_scope('pooling'):
        scale = 1 / 4.0
        h_pool = max_pool_2x2(h_conv)

    with tf.name_scope('convolution2'):
        W_conv2 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM, FILTER_NUM2], name='weight_conv2')
        b_conv2 = bias_variable([FILTER_NUM2], name='bias_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_pool, W_conv2) + b_conv2)

    with tf.name_scope('pooling2'):
        scale /= 4.0
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fully-connected'):
        W_fc = weight_variable([int(IMAGE_SIZE * scale * FILTER_NUM2), FEATURE_DIM], name='weight_fc')
        b_fc = bias_variable([FEATURE_DIM], name='bias_fc')
        h_pool_flat = tf.reshape(h_pool2, [-1, int(IMAGE_SIZE * scale * FILTER_NUM2)])
        h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_drop = tf.nn.dropout(h_fc, keep_prob)

    with tf.name_scope('readout'):
        W = weight_variable([FEATURE_DIM, CATEGORY_NUM], name='weight')
        b = bias_variable([CATEGORY_NUM], name='bias')
        y = tf.nn.softmax(tf.matmul(h_drop, W) + b)

    with tf.name_scope('optimize'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y + 1e-30), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/train', sess.graph)
        # test_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/test', sess.graph)

        # y_hist = tf.histogram_summary("y", y)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy_summary = tf.scalar_summary('accuracy', accuracy)
        # test_accuracy_summary = tf.scalar_summary('accuracy', accuracy)

        loss_summary = tf.scalar_summary("cross_entropy", cross_entropy)

        data_manager = CsvManager()
        data_manager.pre_process()
        data_manager.shuffle_data_and_events()

        sess.run(tf.initialize_all_variables())
        for i in range(data_manager.length):
            if i % SUMMARY_INTERVAL == 0:
                print('step %d' % i)
                if i <= IMAGE_HEIGHT:
                    continue

                data_set = data_manager.get_data_and_events(i, IMAGE_HEIGHT)

                sess.run(train_step, {x: data_set[0], y_: data_set[1], keep_prob: KEEP_PROB})

                print("label")
                print(data_set[1])
                print("y")
                print(sess.run(y, {x: data_set[0], y_: data_set[1], keep_prob: 1.0}))
                print("cross_entropy")
                print(sess.run(cross_entropy, {x: data_set[0], y_: data_set[1], keep_prob: 1.0}))

            
                summary = sess.run(tf.merge_all_summaries(), {x: data_set[0], y_: data_set[1], keep_prob: 1.0})
                train_writer.add_summary(summary, i)
                # summary = sess.run(tf.merge_summary([test_accuracy_summary]), {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                # test_writer.add_summary(summary, i)