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
    source_data[:,3:6] = source_data[:,3:6] * 9.8

    tsource_data = source_data
    #Exchange acc and gyr
    # source_data[:,1:4] ,source_data[:,3:6] = tsource_data[:,3:6],tsource_data[:,1:4]

    '''
    Set parameter
    '''
    setting = Setting.settings()
    setting.Ts = 1.0/128.0

    setting.time_Window_size = 5

    setting.gamma = 1.0

    '''
    Zero-volocity Detector
    '''
    zupt_detector = zupt_test.zupte_test(setting)
    ZUPT1 = zupt_detector.GLRT_Detector(source_data[:,1:7])
    print(source_data[:,1:7])

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

    '''
    Run filter
    '''


    '''
    Show Result.
    '''