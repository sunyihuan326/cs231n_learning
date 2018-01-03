# coding:utf-8
'''
Created on 2018/1/3.

@author: chk01
'''
import numpy as np
from collections import Counter


def trick_1():
    '''
    找array中的众数
    '''
    closest_y = np.array([1, 2, 3, 4, 5, 2, 3, 4, 1, 2, 3, 1, 2, 2, 3])
    c = Counter(closest_y)
    print(c)
    print(c.most_common())
    y_pred = c.most_common(1)[0][0]
    print(y_pred)


# trick_1()
