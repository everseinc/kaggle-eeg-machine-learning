'''
== accuracy ==
0.8335056876938987
'''

import numpy
import random
import csv
import tensorflow as tf
from helper import *
from csv_manager import *

IMAGE_WIDTH = 11
IMAGE_HEIGHT = 200
REAL_IMAGE_HEIGHT = 2000
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
CATEGORY_NUM = 6
LEARNING_RATE = 1.0e-4
FILTER_SIZE = 3
FILTER_SIZE4 = 2
FILTER_NUM = 32
FILTER_NUM2 = 32
FILTER_NUM3 = 64
FILTER_NUM4 = 64
FILTER_NUM5 = 64
FILTER_NUM6 = 64
FEATURE_DIM = 1024
KEEP_PROB = 0.5
TRAINING_LOOP = 260000
BATCH_SIZE = 100
SUMMARY_DIR = 'log_cnn_sl'
RUN_INTERVAL = 10
SUMMARY_INTERVAL = 5000
PRINT_INTERVAL = 1000

with tf.Graph().as_default():
    with tf.name_scope('input'):
        y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM], name='labels')
        x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH], name='input_images')

    with tf.name_scope('convolution'):
        W_conv = weight_variable([FILTER_SIZE, FILTER_SIZE, 1, FILTER_NUM], name='weight_conv')
        b_conv = bias_variable([FILTER_NUM], name='bias_conv')
        x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
        h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)

    with tf.name_scope('convolution2'):
        W_conv2 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM, FILTER_NUM2], name='weight_conv2')
        b_conv2 = bias_variable([FILTER_NUM2], name='bias_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_conv, W_conv2) + b_conv2)

    with tf.name_scope('pooling2'):
        scale = 1 / 4.0
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('normalization2'):
        h_norm2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    with tf.name_scope('convolution3'):
        W_conv3 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM2, FILTER_NUM3], name='weight_conv3')
        b_conv3 = bias_variable([FILTER_NUM3], name='bias_conv3')
        h_conv3 = tf.nn.relu(conv2d(h_norm2, W_conv3) + b_conv3)

    with tf.name_scope('convolution4'):
        W_conv4 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM3, FILTER_NUM4], name='weight_conv4')
        b_conv4 = bias_variable([FILTER_NUM4], name='bias_conv4')
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    with tf.name_scope('normalization4'):
        h_norm4 = tf.nn.lrn(h_conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')

    with tf.name_scope('pooling4'):
        scale /= 4.0
        h_pool4 = max_pool_2x2(h_norm4)

    # with tf.name_scope('convolution5'):
    #     W_conv5 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM4, FILTER_NUM5], name='weight_conv5')
    #     b_conv5 = bias_variable([FILTER_NUM5], name='bias_conv5')
    #     h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

    # with tf.name_scope('convolution6'):
    #     W_conv6 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM5, FILTER_NUM6], name='weight_conv6')
    #     b_conv6 = bias_variable([FILTER_NUM6], name='bias_conv6')
    #     h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    # with tf.name_scope('pooling6'):
    #     scale /= 4.0
    #     h_pool6 = max_pool_2x2(h_conv6)

    with tf.name_scope('fully-connected'):
        W_fc = weight_variable([int(12 * 200 * scale * FILTER_NUM4), FEATURE_DIM], name='weight_fc')
        b_fc = bias_variable([FEATURE_DIM], name='bias_fc')
        h_pool_flat = tf.reshape(h_pool4, [-1, int(12 * 200 * scale * FILTER_NUM4)])
        h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)

    # with tf.name_scope('fully-connected2'):
    #     W_fc2 = weight_variable([FEATURE_DIM, FEATURE_DIM], name='weight_fc2')
    #     b_fc2 = bias_variable([FEATURE_DIM], name='bias_fc2')
    #     h_fc2 = tf.nn.relu(tf.matmul(h_fc, W_fc2) + b_fc2)

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
        train_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/train2_5', sess.graph)
        # test_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/test', sess.graph)

        # y_hist = tf.histogram_summary("y", y)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy_summary = tf.scalar_summary('accuracy', accuracy)
        # test_accuracy_summary = tf.scalar_summary('accuracy', accuracy)

        loss_summary = tf.scalar_summary("cross_entropy", cross_entropy)

        for n in range(1, 8):

            data_manager = CsvManager([7], [n])
            data_manager.pre_process(True, [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 15]) # F7,F3,F4,F8,FC5,FC1,FC2,FC6,Cz
            data_manager.shuffle_data_and_events(REAL_IMAGE_HEIGHT)

            # saver
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state('./saver_cnn_sl/train2_5/')
            if ckpt:
                last_model = ckpt.model_checkpoint_path
                print("load " + last_model)
                saver.restore(sess, last_model)
            else:
                sess.run(tf.initialize_all_variables())

            count = 0
            correct_count = 0

            for i in range(data_manager.length - REAL_IMAGE_HEIGHT):
                if i % RUN_INTERVAL == 0:
                    data_set = data_manager.get_data_and_events(i, IMAGE_HEIGHT, REAL_IMAGE_HEIGHT)

                    sess.run(train_step, {x: data_set[0], y_: data_set[1], keep_prob: KEEP_PROB})

                    count += 1
                    if sess.run(accuracy, {x: data_set[0], y_: data_set[1], keep_prob: 1.0}) == 1:
                        correct_count += 1

                if i % SUMMARY_INTERVAL == 0:
                    summary = sess.run(tf.merge_all_summaries(), {x: data_set[0], y_: data_set[1], keep_prob: 1.0})
                    train_writer.add_summary(summary, i)
                    # summary = sess.run(tf.merge_summary([test_accuracy_summary]), {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                    # test_writer.add_summary(summary, i)

                if i % PRINT_INTERVAL == 0:
                    print('step %d' % i)
                    print("label")
                    print(data_set[1])
                    print("y")
                    print(sess.run(y, {x: data_set[0], y_: data_set[1], keep_prob: 1.0}))
                    # print('h_pool6')
                    # print(sess.run(h_pool6, {x: data_set[0], y_: data_set[1], keep_prob: 1.0}))
                    print("cross_entropy")
                    print(sess.run(cross_entropy, {x: data_set[0], y_: data_set[1], keep_prob: 1.0}))


            saver.save(sess, "./saver_cnn_sl/train2_5/model.ckpt")
            print('== accuracy ==')
            print(correct_count / count)