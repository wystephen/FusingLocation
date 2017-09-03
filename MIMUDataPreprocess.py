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

    out_data[:,1:7] = src_data[:,3:9]

    # out_data[:,3] *= -1.0
    # out_data[:,7] *= -1.0
    out_data[:,4:7] /= 180.0*np.pi


    np.savetxt("/home/steve/XsensData/1Imu.csv",out_data,delimiter=',')

    t_setting = OPENSHOE.Setting.settings()
    # t_setting.Ts = 1.0/400.0
    t_setting.Ts = 1.0/100.0

    t_setting.sigma_a /= 1.5
    t_setting.sigma_g /= 1.5

    t_setting.sigma_acc *= 1.3
    t_setting.sigma_gyro *= 1.3

    t_setting.init_heading = t_setting.init_heading1 = t_setting.init_heading2 = 0.0



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

    ins_filter = OPENSHOE.PdrEkf.ZUPTaidedIns(settings=t_setting)
    ins_filter.init_Nav_eq(out_data[1:40,1:7],out_data[1:40,1:7])

    u1 = out_data[:,1:7]
    u2 = u1

    zupt1 = zupt2 = zupt_result

    all_x = np.zeros([18,out_data.shape[0]])

    for index in range(u1.shape[0]):
        if(index % 100 ==0):
            print("finished : " + str(float(index) / u1.shape[0]))
        # if index > 1:
        #     ins_filter.para.TS = out_data[index,0]-out_data[index-1,0]
        all_x[:,index] = ins_filter.GetPosition(
            u1[index,:],
            u2[index,:],
            zupt1[index,:],
            zupt2[index,:]
        ).reshape([18])

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.plot(all_x[0,:],all_x[1,:],all_x[2,:])
    # plt.show()


    plt.show()
