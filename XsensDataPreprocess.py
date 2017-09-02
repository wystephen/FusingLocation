# -*- coding:utf-8 -*-
# Created by steve @ 17-9-2 上午10:53


import numpy as np
import matplotlib.pyplot as plt

import OPENSHOE.zupt_test
import OPENSHOE.Setting


if __name__ == '__main__':
    src_data = np.loadtxt('/home/steve/XsensData/1.csv', delimiter=',')

    out_data = src_data

    out_data[:, 0] /= 400.0
    out_data[:, 4:7] = out_data[:, 4:7] / 180.0 * np.pi

    np.savetxt("/home/steve/XsensData/1Imu.csv",out_data,delimiter=',')

    t_setting = OPENSHOE.Setting.settings()

    zupt_detector = OPENSHOE.zupt_test.zupte_test(t_setting)

    zupt_result = zupt_detector.GLRT_Detector(out_data[:,1:7])
    np.savetxt("/home/steve/XsensData/1Zupt.csv",zupt_result,delimiter=',')


    plt.figure()
    plt.plot(out_data[:, 1:4])
    plt.plot(zupt_result)

    plt.figure()
    plt.plot(out_data[:, 4:7])
    plt.plot(zupt_result)

    '''
    Test EKF
    '''

    ins



    plt.show()
