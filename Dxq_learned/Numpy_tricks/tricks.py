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
    随机选择数组中的samples_per_class个
    replace：是否放回 False|True
    p：概率[.3,.2,.5]
    :return:
    '''
    idxs = [1, 4, 8, 10]
    samples_per_class = 2
    idxs = np.random.choice(idxs, samples_per_class, replace=False, p=[0.001, 0.001, 0.298, 0.7])  # 不重复,每个数字出现的概率
    print(idxs)


trick_2()
