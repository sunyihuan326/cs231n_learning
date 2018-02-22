# coding:utf-8
'''
Created on 2018/2/22.

@author: chk01
'''
import tensorflow as tf


def trick_1():
    '''
    合并和分解矩阵
    :return:
    '''
    a = tf.constant(value=[1, 2, 3, 4])
    b = tf.constant(value=[2, 3, 4, 5])
    b2 = tf.constant(value=[2, 3, 4, 5])
    c = tf.stack([a, b, b2], axis=0)
    d = tf.stack([a, b, b2], axis=1)
    print(c)
    e = tf.unstack(c, axis=0)
    f = tf.unstack(c, axis=1)
    with tf.Session() as sess:
        _c, _d, _e, _f = sess.run([c, d, e, f])
        print(_c)
        print(type(_c))
        print(_d)
        print(_e)
        print(type(_e))
        print(_f)


def trick_2():
    '''
    输出选中位置上的数值
    位置：[1,3] 不能出现-1
    数值：[2,4]
    '''
    a = [[0, 4, 8, 12], [1, 5], [2], [3]]
    # a2 = [[0, 1, 2], [3, 4, 5], [6, 7], [9, 10], [11]]
    b = tf.nn.embedding_lookup(a, [0, 1, 2, 3, 4, 5])

    with tf.Session() as sess:
        _b = sess.run(b)
        print(_b)


trick_2()
