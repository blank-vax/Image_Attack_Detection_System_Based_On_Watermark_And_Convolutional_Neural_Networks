# -*- coding: utf-8 -*-

import glob
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from matplotlib import pyplot as plt
import threading
import time
import math

# inception-v3 bottleneck layer numbers
BOTTLENECT_TENSOR_SIZE = 2048

# Inception-v3 bottleneck layer result tensor name
BOTTLENECT_TENSOR_NAME = 'pool_3/_reshape:0'

# Input image tensor name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# Inception-v3 model path
# You could change it to your own path
MODEL_DIR = model_path

# Inception-v3 model file name
MODEL_FILE = 'tensorflow_inception_graph.pb'

# Cache dir
# You could change it to your own path
CACHE_DIR = cache_path

# Input image dir
# You could change it to your own path
INPUT_DATA = image_path

# Validation percentage
VALIDATION_PRECENTAGE = 10

# Testing percentage
TEST_PRECENTAGE = 10

# paras
BASE_LEARNING_RATE = 0.1
LEARNING_RATE = 0.1
MIN_LEARNING_RATE = 0.01
STEPS = 1550
BATCH = 100
NUM_CHECKPOINTS = 5
CHECKPOINT_EVERY = 50

# for viewer
result = {}


# Split image data
def create_image_lists(testing_percentage, validation_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }
    return result


# Get a specific image path
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# After calculating by Inception-v3 model,the image tensor path
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index,
                          category) + '.txt'


# Processing with Inception...
def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# save data
def get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                             jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          category)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index,
                                    category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


# random data for testing
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many,
                                  category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category,
            jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


# Validating accuracy by testing dataset
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor,
                         bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(
                image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index, category,
                jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    global LEARNING_RATE
    image_lists = create_image_lists(TEST_PRECENTAGE, VALIDATION_PRECENTAGE)
    n_classes = len(image_lists.keys())  
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
            graph_def,
            return_elements=[BOTTLENECT_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        bottleneck_input = tf.placeholder(
            tf.float32, [None, BOTTLENECT_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(
            tf.float32, [None, n_classes], name='GroundTruthInput')

        # full-connection layer for classifying
        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [BOTTLENECT_TENSOR_SIZE, n_classes], stddev=0.001))
            biases = tf.Variable(tf.zeros([n_classes]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
            cross_entropy_mean)

        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(
                tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            evaluation_step = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            init = tf.global_variables_initializer().run()

            # The save_path of model and summary
            import time
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(
                os.path.join(os.path.curdir, 'runs', timestamp))
            print('\nWriting to {}\n'.format(out_dir))
            # The summary of lost value and accurate rate
            loss_summary = tf.summary.scalar('loss', cross_entropy_mean)
            acc_summary = tf.summary.scalar('accuracy', evaluation_step)
            # Train summary
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                        sess.graph)
            # Develop summary
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            # Save the checkpoint
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(
                    tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)
            
            for i in range(STEPS):
                LEARNING_RATE = BASE_LEARNING_RATE / (math.exp(
                    i / (STEPS / math.log(0.1 / MIN_LEARNING_RATE))))
                train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks \
                    (sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
                _, train_summaries = sess.run(
                    [train_step, train_summary_op],
                    feed_dict={
                        bottleneck_input: train_bottlenecks,
                        ground_truth_input: train_ground_truth
                    })

                # Save the summary of each step
                train_summary_writer.add_summary(train_summaries, i)

                # calculating accuracy on validation dataset
                if i % 10 == 0 or i + 1 == STEPS:
                    validation_bottlenecks, validation_ground_truth = \
                        get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH,
                                                      'validation', jpeg_data_tensor, bottleneck_tensor)
                    validation_accuracy, dev_summaries = sess.run(
                        [evaluation_step, dev_summary_op],
                        feed_dict={
                            bottleneck_input: validation_bottlenecks,
                            ground_truth_input: validation_ground_truth
                        })
                    global result
                    result['{:0>4d}'.format(i)] = validation_accuracy
                    print(
                        'Step %d :Validation accuracy on random sampled %d examples = %.1f%%'
                        % (i, BATCH, validation_accuracy * 100))

                # Save model and test summary per checkpoint
                if i % CHECKPOINT_EVERY == 0:
                    dev_summary_writer.add_summary(dev_summaries, i)
                    path = saver.save(sess, checkpoint_prefix, global_step=i)
                    print('Saved model checkpoint to {}\n'.format(path))

            # test final accuracy
            test_bottlenecks, test_ground_truth = get_test_bottlenecks(
                sess, image_lists, n_classes, jpeg_data_tensor,
                bottleneck_tensor)
            test_accuracy = sess.run(
                evaluation_step,
                feed_dict={
                    bottleneck_input: test_bottlenecks,
                    ground_truth_input: test_ground_truth
                })
            print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

            # Save labels
            output_labels = os.path.join(out_dir, 'labels.txt')
            with tf.gfile.FastGFile(output_labels, 'w') as f:
                keys = list(image_lists.keys())
                for i in range(len(keys)):
                    keys[i] = '%2d -> %s' % (i, keys[i])
                f.write('\n'.join(keys) + '\n')


# plotter for accuracy
def polter():
    plt.ion()
    plt.style.use('dark_background')
    plt.figure()
    while True:
        global result
        data = result
        V = [key for key in data.values()]
        K = [key for key in data.keys()]
        if len(V) < 1:
            time.sleep(5)
            continue
        plt.plot(V)
        plt.xticks(rotation=60)
        plt.xticks(range(len(V)), K)
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy--Training Steps Figure")
        plt.pause(2)


if __name__ == '__main__':
    threading.Thread(target=polter).start()
    tf.app.run()