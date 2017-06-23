import argparse
import logging
import tensorflow as tf
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

from utils import *

def shallow_features_softmax():
    # Based on: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py
    BATCH_SIZE = 128
    ds = get_cifar('train', BATCH_SIZE)

    # Create the model
    n_features = 32 * 32 * 3
    n_classes = 10
    x = tf.placeholder(tf.float32, [None, n_features])
    W = tf.Variable(tf.zeros([n_features, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.int64, [None])
    entr = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(entr)
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    # Train the model
    for d in ds:
        d[0] = np.reshape(d[0], [BATCH_SIZE, n_features])
        sess.run(train_step, feed_dict={x: d[0], y_: d[1]})

    # Test the model
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    N_TEST_IMGS = 1000
    ds = get_cifar('test', N_TEST_IMGS) # all test examples at once
    images, labels = next(ds)
    images = np.reshape(images, [N_TEST_IMGS, n_features]) # [1000, 32, 32, 3] -> [1000, 32 * 32 * 3]
    acc = sess.run(accuracy, feed_dict={x: images, y_: labels})

    return acc


def shallow_features_svm(use_hog=False, kernelize=False):
    logging.info('Reading the training data...')
    ds = get_cifar('train')
    images = []
    labels = []
    n_features = 32 * 32 * 3
    for d in ds:
        if use_hog:
            img = color.rgb2gray(d[0])
            img = hog(img, orientations=8, pixels_per_cell=(8, 8))
        else:
            img = np.reshape(d[0], [n_features])
        images.append(img)
        labels.append(d[1])
    hog_images = np.vstack(images)

    logging.info('Training SVM classifier on {} examples...'.format(len(labels)))
    if kernelize:
        clf = SVC(verbose=True).fit(hog_images, labels) 
    else:
        clf = LinearSVC(verbose=True).fit(hog_images, labels) 

    # Get test data
    ds = get_cifar('test')
    test_images = []
    test_labels = []
    logging.info('Loading test data...')
    for d in ds:
        if use_hog:
            img = color.rgb2gray(d[0])
            img = hog(img, orientations=8, pixels_per_cell=(8, 8))
        else:
            img = np.reshape(d[0], [n_features])
        test_images.append(img)
        test_labels.append(d[1])
    logging.info('Testing the SVM classifier on {} examples...'.format(len(test_labels)))
    pred_labels = clf.predict(test_images)
    acc = accuracy_score(test_labels, pred_labels)
    return acc


def cnn_features_svm(kernelize=True):
    logging.info('Loading CNN codes...')
    data_train = np.load('cnncodes_train.npz')
    data_test = np.load('cnncodes_test.npz')
    X_train, y_train = data_train['cnn_codes'], data_train['y']
    X_test, y_test = data_test['cnn_codes'], data_test['y']

    logging.info('Training SVM on {} examples...'.format(len(y_train)))
    C = 0.1
    if kernelize:
        svc = SVC(verbose=True, C=C).fit(X_train,y_train)
    else:
        svc = LinearSVC(verbose=True, C=C).fit(X_train,y_train)
    
    logging.info('Testing SVM on {} examples...'.format(len(y_test)))
    y_test_pred = svc.predict(X_test)
    acc = np.sum(y_test_pred == y_test)/len(y_test)
    return acc


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    modes = ['softmax_raw', 'svm_raw', 'svm_hog', 
             'svm_hog_kern', 'svm_cnn', 'svm_cnn_kern']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=modes[0], choices=modes,
                      help='Operation to perform. Possible values: {}'.format(', '.join(modes)))
    args = parser.parse_args()

    if args.mode == 'softmax_raw':
        acc = shallow_features_softmax()
        logging.info('Test accuracy for softmax classifier on raw images: {:.1f}%'.format(acc * 100))

    elif args.mode == 'svm_raw':
        acc = shallow_features_svm()
        logging.info('Test accuracy for SVM classifier on raw images: {:.1f}%'.format(acc * 100))

    elif args.mode == 'svm_hog':
        acc = shallow_features_svm(use_hog = True)
        logging.info('Test accuracy for SVM classifier on HOG features: {:.1f}%'.format(acc * 100))

    elif args.mode == 'svm_hog_kern':
        acc = shallow_features_svm(use_hog = True, kernelize=True)
        logging.info('Test accuracy for SVM classifier with kernel trick on HOG features: {:.1f}%'.format(acc * 100))

    elif args.mode == 'svm_cnn':
        acc = cnn_features_svm()
        logging.info('Test accuracy for SVM classifier on CNN codes: {:.1f}%'.format(acc * 100))

    elif args.mode == 'svm_cnn_kern':
        acc = cnn_features_svm(kernelize=True)
        logging.info('Test accuracy for kernelized SVM classifier on CNN codes: {:.1f}%'.format(acc * 100))

    else:
        logging.warning('Uknown mode. Possible values: {}'.format(', '.join(modes)))