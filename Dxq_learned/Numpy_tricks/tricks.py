# coding:utf-8
'''
Created on 2018/1/3.

@author: chk01
'''
import numpy as np


def trick_1():
    '''
    寻找所有label==y的index
    '''
    y_train = np.array([1, 2, 3, 4, 2, 3, 4, 4, 2, 1, 2, 3, 4, 4])
    y = 2
    idxs = np.flatnonzero(y_train == y)
    print(idxs)


def trick_2():
    '''
    随机选择array中的samples_per_class个
    replace：是否放回 False|True
    p：概率[.3,.2,.5]
    :return:
    '''
    idxs = np.array([1, 4, 8, 10])
    samples_per_class = 2
    idxs = np.random.choice(idxs, samples_per_class, replace=False, p=[0.25, 0.25, 0.25, 0.25])  # 不重复,每个数字出现的概率
    print(idxs)


def trick_3():
    '''
    寻找数组中的Top-k的index
    '''
    k = 1
    idxs = np.array([1, 4, 8, 10, 2, 3])
    sorted_list = np.argsort(idxs)
    top_k_index = sorted_list[:k]
    top_k_value = idxs[top_k_index]
    print(top_k_index)
    print(top_k_value)


def trick_4():
    '''
    分割数据集
    '''
    num_folds = 4
    X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    # 打乱
    np.random.shuffle(X_train)

    split_arr = np.array_split(X_train, num_folds)
    for i in range(num_folds):
        print(split_arr[i])


def trick_5():
    '''
    扩展矩阵broadcast
    '''
    a = np.array([1, 2]).reshape(-1, 1)
    b = np.array([3, 4, 5]).reshape(1, -1)
    print(a.shape)
    print(b.shape)
    print(a + b)


def trick_6():
    '''
    合并数据
    '''
    num_folds = 4
    X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    Y_train = [1, 2, 3, 4, 5]
    # np.random.shuffle(X_train)
    # np.random.shuffle(Y_train)
    # 按顺序的
    split_arr_X = np.array_split(X_train, num_folds)
    split_arr_Y = np.array_split(Y_train, num_folds)
    j = 2
    print(split_arr_X[j])
    print(split_arr_Y[j])
    # 剔除J行数据
    # 垂直合并 增加m
    print(split_arr_X[0:j] + split_arr_X[j + 1:])
    X_train_cv = np.vstack(split_arr_X[0:j] + split_arr_X[j + 1:])
    # np.concatenate(split_arr_X[0:j] + split_arr_X[j + 1:], axis=0)
    print(split_arr_Y[0:j] + split_arr_Y[j + 1:])
    y_train_cv = np.hstack(split_arr_Y[0:j] + split_arr_Y[j + 1:])
    # np.concatenate(split_arr_Y[0:j] + split_arr_Y[j + 1:], axis=0)
    print(X_train_cv)
    print(y_train_cv)


def trick_7():
    '''
    乱序
    '''

    x = [1, 2, 3, 4]
    y = np.random.permutation(x)
    print(y)


trick_7()
