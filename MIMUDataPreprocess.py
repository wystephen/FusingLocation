# -*- coding:utf-8 -*-
# Created by steve @ 17-9-3 上午11:14


import numpy as np
import matplotlib.pyplot as plt

import OPENSHOE.zupt_test
import OPENSHOE.PdrEkf
import OPENSHOE.Setting

import mpl_toolkits.mplot3d.axes3d as p3


if __name__ == '__main__':
    src_data = np.loadtxt('/home/steve/XsensData/test.csv',delimiter=',')

    out_data = np.zeros([src_data.shape[0],10])

    out_data[:,1:3] =
