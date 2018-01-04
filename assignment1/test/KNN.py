# coding:utf-8
'''
Created on 2018/1/2

@author: sunyihuan
'''
import random
import numpy as np
import matplotlib.pyplot as plt
from assignment1.data_utils import load_CIFAR10
from assignment1.classifiers import KNearestNeighbor
import scipy.io as scio
import os

# cifar10_dir = '../datasets/cifar-10-batches-py'
# X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)


# As a sanity check, we print out the size of the training and test data.
# Training data shape:  (50000, 32, 32, 3)
# Training labels shape:  (50000,)
# Test data shape:  (10000, 32, 32, 3)
# Test labels shape:  (10000,)

def visualize_examples():
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.savefig('../res/visual_example.jpg')


def preprocessing(X_train, y_train, X_test, y_test):
    num_training = 5000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    return X_train, y_train, X_test, y_test


def cal_dists(X_train, y_train, X_test, y_test):
    X_train, y_train, X_test, y_test = preprocessing(X_train, y_train, X_test, y_test)
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    if not os.path.exists('../res/dists'):
        X_train, y_train, X_test, y_test = preprocessing(X_train, y_train, X_test, y_test)
        dists = classifier.compute_distances_two_loops(X_test)
        scio.savemat('../res/dists', {"data": dists})
    dists = scio.loadmat('../res/dists')['data']
    y_test_pred = classifier.predict_labels(dists, k=5)
    # Compute and print the fraction of correctly predicted examples
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / 500
    print('Got %d / %d correct => accuracy: %f' % (num_correct, 500, accuracy))


if __name__ == '__main__':
    # visualize_examples()
    # cal_dists(X_train, y_train, X_test, y_test)
    # plt.imshow(dists, interpolation='none')
    # plt.savefig('../res/dists.jpg')

    dists = scio.loadmat('../res/dists')['data']
    dists_one = scio.loadmat('../res/dists_one')['data']
    difference = np.linalg.norm(dists - dists_one, ord='fro')
    print('Difference was: %f' % (difference,))
    if difference < 0.001:
        print('Good! The distance matrices are the same')
    else:
        print('Uh-oh! The distance matrices are different')