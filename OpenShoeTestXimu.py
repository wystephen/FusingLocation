# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 02　18:52

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from OPENSHOE import PdrEkf,Setting,zupt_test

import XimuDataPreProcess


if __name__ == '__main__':

    '''
    Load Data
    '''
    xdpp = XimuDataPreProcess.XimuDataPreProcess("test5")

    source_data = xdpp.data_index


    '''
    PreProcessData
    '''

    #deg to rad
    source_data[:,1:4] = source_data[:,1:4] * np.pi / 180.0

    # g to m/s^2
    source_data[:,4:7] = source_data[:,4:7] * 9.80

    tsource_data = source_data
    #Exchange acc and gyr
    source_data[:,1:4] ,source_data[:,4:7] = tsource_data[:,4:7],tsource_data[:,1:4]

    '''
    Set parameter
    '''
    setting = Setting.settings()
    # setting.Ts = 1.0/128.0

    setting.Ts = 0.025974
    setting.min_rud_sep = int(1/setting.Ts)

    setting.time_Window_size = 5

    setting.range_constraint_on = False

    # setting.gamma = 1.0

    '''
    Zero-volocity Detector
    '''
    zupt_detector = zupt_test.zupte_test(setting)
    ZUPT1 = zupt_detector.GLRT_Detector(source_data[:,1:7])
    # print(source_data[:,1:7])
    #
    plt.figure(1)
    plt.plot(ZUPT1*100,'r-')
    plt.plot(source_data[:,1],'g-+')
    plt.plot(source_data[:,2],'y-+')
    plt.plot(source_data[:,3],'b-+')

    plt.grid(True)

    plt.show()



    '''
    Initial EKF
    '''
    ins_filter = PdrEkf.ZUPTaidedIns(setting)

    ins_filter.init_Nav_eq(source_data[1:20,1:],source_data[1:20,1:])

    '''
    Run filter
    '''
    u1 = source_data[:,1:]
    u2 = source_data[:,1:]

    zupt1 = zupt2 = ZUPT1

    all_x = np.zeros([18,source_data.shape[0]])

    for index in range(u1.shape[0]):
        if (index % 100 == 0):
            print(float(index) / u1.shape[0])
        all_x[:, index] = ins_filter.GetPosition(u1[index, :],
                                                 u2[index, :],
                                                 zupt1[index],
                                                 zupt2[index]).reshape([18])


    '''
    Show Result.
    '''
    print(np.linalg.norm(all_x[0:3, u1.shape[0] - 1]))
    print(np.linalg.norm(all_x[9:12, u1.shape[0] - 1]))

    # SHOW RESULT
    plt.figure(1)
    plt.grid()

    # plt.plot(all_x[0, :], all_x[1, :], all_x[2, :], 'r')
    # plt.plot(all_x[9, :], all_x[10, :], all_x[11, :], 'b')

    plt.plot(all_x[0, :], all_x[1, :], 'r+')

    plt.figure(2)
    plt.plot(all_x[2,:],'y-+')

    # plt.figure(12)
    #
    # plt.plot(all_x[3, :], 'r')
    # plt.plot(all_x[4, :], 'g')
    # plt.plot(all_x[5, :], 'b')
    #
    # plt.figure(13)
    # plt.plot(u1[:, 0], 'r')
    # plt.plot(u1[:, 1], 'g')
    # plt.plot(u1[:, 2], 'b')
    #
    # plt.plot(zupt1[:] * 10.0, 'y')
    #
    # plt.figure(14)
    #
    # plt.plot(all_x[6, :], 'r')
    # plt.plot(all_x[7, :], 'g')
    # plt.plot(all_x[8, :], 'b')
    #
    # plt.figure(15)
    #
    # plt.plot(u1[:, 3], 'r')
    # plt.plot(u1[:, 4], 'g')
    # plt.plot(u1[:, 5], 'b')
    #
    # plt.figure(22)
    # plt.plot(all_x[12, :], 'r')
    # plt.plot(all_x[13, :], 'g')
    # plt.plot(all_x[14, :], 'b')
    #
    # plt.figure(23)
    # plt.plot(u2[:, 0], 'r')
    # plt.plot(u2[:, 1], 'g')
    # plt.plot(u2[:, 2], 'b')
    #
    # plt.plot(zupt2[:] * 10.0, 'y')
    #
    # plt.figure(24)
    #
    # plt.plot(all_x[15, :], 'r')
    # plt.plot(all_x[16, :], 'g')
    # plt.plot(all_x[17, :], 'b')
    #
    # plt.figure(25)
    #
    # plt.plot(u2[:, 3], 'r')
    # plt.plot(u2[:, 4], 'g')
    # plt.plot(u2[:, 5], 'b')

    plt.show()