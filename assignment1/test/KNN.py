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

cifar10_dir = '../datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)


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


def cross_validation(X_train, y_train, X_test, y_test):
    X_train, y_train, X_test, y_test = preprocessing(X_train, y_train, X_test, y_test)
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    k_to_accuracies = {}
    for k in k_choices:  # find the best k-value
        k_to_accuracies[k] = []
        for i in range(num_folds):
            X_train_cv = np.concatenate(X_train_folds[:i] + X_train_folds[i + 1:])
            X_test_cv = X_train_folds[i]

            y_train_cv = np.concatenate(y_train_folds[:i] + y_train_folds[i + 1:])  # size:4000
            y_test_cv = y_train_folds[i]

            classifier.train(X_train_cv, y_train_cv)
            dists_cv = classifier.compute_distances_no_loops(X_test_cv)

            y_test_pred = classifier.predict_labels(dists_cv, k)
            num_correct = np.sum(y_test_pred == y_test_cv)
            accuracy = float(num_correct) / y_test_cv.shape[0]

            k_to_accuracies[k].append(accuracy)
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


if __name__ == '__main__':
    # visualize_examples()
    # cal_dists(X_train, y_train, X_test, y_test)
    # plt.imshow(dists, interpolation='none')
    # plt.savefig('../res/dists.jpg')

    cross_validation(X_train, y_train, X_test, y_test)
