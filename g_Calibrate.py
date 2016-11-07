# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 07　10:13

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from    scipy.optimize import minimize

'''
This class is use to compute the transform matrix to calibration the ximu use data in pdr.
'''
class gCalibration:
    def __init__(self,ZUPT,source_data):
        '''
        Initial,use zupt and source data.
        :param ZUPT:
        :param source_data:
        '''
        self.source_data = source_data

        self.zupt = ZUPT

    def SelectData(self):
        data_size = np.sum(self.zupt)
        print(data_size)
        self.zerovdata = np.zeros([int(data_size),3])
        the_index = 0

        for i in range(self.zupt.shape[0]):
            if self.zupt[i] > 0.6:
                '''
                '''
                self.zerovdata[the_index,:] = self.source_data[i,1:4]
                the_index += 1

    def CostFunction(self,theta_acc):
        '''

        :param theta_acc:
        :return:
        '''
        ka,ta,ba = self.Theta2Matrix(theta_acc)

        tmp_data = np.zeros_like(self.zerovdata)











    def Theta2Matrix(self,theta):
        ka = np.diag(theta[0:3])

        ta = np.zeros([1,1,1])
        ta[0,1] = - theta[3]
        ta[0,2] = theta[5]
        ta[1,2] = theta[4]

        ba = theta[6:9]

        return ka,ta,ba









if __name__ == '__main__':
    from OPENSHOE import zupt_test,Setting

    import XimuDataPreProcess

    xdpp = XimuDataPreProcess.XimuDataPreProcess("test23")

    source_data = xdpp.data_index

    '''
    PreProcessData
    '''

    print("first mean:", np.mean(source_data[:, 1:7], axis=0))
    # deg to rad
    source_data[:, 1:4] = source_data[:, 1:4] * np.pi / 180.0

    # g to m/s^2
    source_data[:, 4:7] = source_data[:, 4:7] * 9.80

    tsource_data = source_data.copy()
    # Exchange acc and gyr
    # source_data[:,1:4] ,source_data[:,4:7] = tsource_data[:,4:7],tsource_data[:,1:4]
    source_data[:, 1:4] = tsource_data[:, 4:7]
    source_data[:, 4:7] = tsource_data[:, 1:4]

    print("mean:", np.mean(source_data[:, 1:7], axis=0))
    '''
    Set parameter
    '''
    setting = Setting.settings()
    # setting.Ts = 1.0/128.0

    setting.Ts = np.mean(source_data[1:, 0] - source_data[0:-1, 0])
    print("setting:", setting.Ts)
    setting.min_rud_sep = int(1 / setting.Ts)
    setting.range_constraint_on = False

    # For Zero Velocity Detector
    setting.time_Window_size = 5
    setting.gamma = 6580
    # setting.sigma_a = 0.05
    # setting.sigma_g = 0.35 * np.pi / 180.0

    # For Ekf Filter
    setting.init_heading1 = setting.init_heading2

    # setting.sigma_acc = setting.sigma_acc / 2.0
    # setting.sigma_gyro = 4.0 * 10.0 * np.ones([3,1]) * 0.0 * np.pi / 180.0


    '''
    Zero-volocity Detector
    '''
    zupt_detector = zupt_test.zupte_test(setting)
    ZUPT1 = zupt_detector.GLRT_Detector(source_data[:, 1:7])

    '''
    calibration
    '''

    gca = gCalibration(ZUPT1,source_data)
    gca.SelectData()

    print(gca.zerovdata.shape)
    plt.figure(1)
    plt.plot(gca.zerovdata[:,0],'r-+')
    plt.plot(gca.zerovdata[:,1],'g-+')
    plt.plot(gca.zerovdata[:,2],'b-+')

    plt.show()

