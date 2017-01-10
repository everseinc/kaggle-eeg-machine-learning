import numpy
import tensorflow as tf
from csv_manager import *
from helper import *

IMAGE_WIDTH = 1
IMAGE_HEIGHT = 32
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
CATEGORY_NUM = 7
LEARNING_RATE = 0.001
FEATURE_DIM = 100
TRAINING_LOOP = 622000
BATCH_SIZE = 100
SUMMARY_DIR = 'log_softmax_fc'
SUMMARY_INTERVAL = 100

# mnist = input_data.read_data_sets('data', one_hot=True)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM], name='labels')
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name='input_images')

    with tf.name_scope('fully-connected'):
        W_fc_t = tf.truncated_normal([IMAGE_SIZE, FEATURE_DIM], stddev=0.1)
        W_fc = tf.Variable(W_fc_t, name='weight_fc')
        b_fc_t = tf.constant(0.1, shape=[FEATURE_DIM])
        b_fc = tf.Variable(b_fc_t, name='bias_fc')
        test = tf.truncated_normal([IMAGE_SIZE, FEATURE_DIM], stddev=0.1)
        test2 = tf.Variable(test, name='test2')
        test3 = tf.constant(0.1, shape=[FEATURE_DIM])
        test4 = tf.Variable(test3, name='test4')
        h_fc = tf.nn.relu(tf.matmul(x, W_fc) + b_fc)

    with tf.name_scope('readout'):
        W_t = tf.truncated_normal([FEATURE_DIM, CATEGORY_NUM], stddev=0.1)
        W = tf.Variable(W_t, name='weight')
        b_t = tf.constant(0.1, shape=[CATEGORY_NUM])
        b = tf.Variable(b_t, name='bias')
        y = tf.nn.softmax(tf.matmul(h_fc, W) + b)

    with tf.name_scope('optimize'):
        hi = y_ * tf.log(y)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/train', sess.graph)

        train_entropy_summary = tf.scalar_summary('cross_entropy', cross_entropy)

        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # train_accuracy_summary = tf.scalar_summary('accuracy', accuracy)

        data_file = open('../EEG_grasp_and_left_data/train/subj10_all_data.csv', 'r')
        label_file = open('../EEG_grasp_and_left_data/train/subj10_all_events.csv', 'r')
        data_reader = csv.reader(data_file)
        label_reader = csv.reader(label_file)

        sess.run(tf.initialize_all_variables())
        for i in range(TRAINING_LOOP + 1):
            data_val = next(data_reader)
            label_val = next(label_reader)
            data_val.pop(0)
            label_val.pop(0)
            data_val_float = [(float(str) + 1000) / 2000 for str in data_val]
            label_val_float = [float(str) for str in label_val]
            data = [data_val_float]
            if sum(label_val_float) == 0:
                label_val_float.insert(0, 1.0)
            else:
                label_val_float.insert(0, 0.0)
            label = [label_val_float]
            # data_queue = tf.train.string_input_producer(["../EEG_grasp_and_left_data/train/subj10_series1_data.csv"])
            # label_queue = tf.train.string_input_producer(["../EEG_grasp_and_left_data/train/subj10_series1_events.csv"])
            # b_data = get_data_from(data_queue)
            # b_label = get_label_from(label_queue)
            # print(b_data)
            # print(b_label)
            sess.run(train_step, {x: data, y_: label})

            if i % SUMMARY_INTERVAL == 0:
                # result = sess.run([merged], {x: data, y_: label})
                # summary_str = result[0]
                # acc = result[1]
                # writer.add_summary(summary_str, i)
                print("step = {0}".format(i))
                summary = sess.run(tf.merge_summary([train_entropy_summary]), {x: data, y_: label})
                train_writer.add_summary(summary, i)
                # print(data)
                # print(label)
                # print("step = {0} test = {1} test2 = {2}".format(i, sess.run(test, {x: data, y_: label}), sess.run(test2, {x: data, y_: label})))
                print(label)
                print("y = {0}".format(sess.run(y, {x: data, y_: label})))
                print("step = {0} hi = {1} cross_entropy = {2}".format(i, sess.run(hi, {x: data, y_: label}), sess.run(cross_entropy, {x: data, y_: label})))
                # summary = sess.run(tf.merge_summary([test_accuracy_summary]), {x: mnist.test.images, y_: mnist.test.labels})
                # test_writer.add_summary(summary, i)

        data_file.close()
        label_file.close()

