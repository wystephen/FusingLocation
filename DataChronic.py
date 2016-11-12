# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 09　20:57

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from OPENSHOE import PdrEkf, Setting, zupt_test

import XimuDataPreProcess

from log_process import seq_process


class DataChronic:
    def __init__(self, dir_name):
        '''
        Load and preprocess data.

        :param dir_name:
        '''

        '''
        Load imu data
        '''
        xdpp = XimuDataPreProcess.XimuDataPreProcess(dir_name)

        self.ImuSourceData = xdpp.data_index
        # print('MARK0',self.ImuSourceData[:,0])

        tmp_data = self.ImuSourceData.copy()

        self.ImuSourceData[:, 1:4] = tmp_data[:, 4:7] * 9.8
        self.ImuSourceData[:, 4:7] = tmp_data[:, 1:4] * np.pi / 180.0

        '''
        Load uwb data
        '''
        se = seq_process()

        import os
        for file_name in os.listdir(dir_name):
            if '.data' in file_name:
                # print("file__name :", file_name)
                se.process_file(file_name=dir_name + '/' + file_name)

        self.UwbData = np.loadtxt('atrange.txt')
        self.BeaconSet = np.loadtxt(dir_name + '/' + 'beaconset')

    def RunOpenshoe(self):
        '''
        Run Open shoe ,get path compute by imu.
        :return:
        '''
        self.openshoeresult = np.zeros([self.ImuSourceData.shape[0], 19])

        '''
        Run Openshoe output the result.
        '''
        setting = Setting.settings()

        setting.Ts = np.mean(self.ImuSourceData[1:, 0] - self.ImuSourceData[0:-1, 0])
        print("Ts:", setting.Ts)

        setting.min_rud_sep = int(1 / setting.Ts)

        setting.time_Window_size = 5
        setting.gamma = 6580

        setting.init_heading2 = setting.init_heading1

        zupt_detector = zupt_test.zupte_test(setting)

        zupt1 = zupt_detector.GLRT_Detector(self.ImuSourceData[:, 1:7])

        # print("MARK1",self.ImuSourceData[:,0])

        ins_filter = PdrEkf.ZUPTaidedIns(setting)

        ins_filter.init_Nav_eq(self.ImuSourceData[1:40, 1:7],
                               self.ImuSourceData[1:40, 1:7])

        for index in range(self.ImuSourceData.shape[0]):
            if (index % 100 == 0):
                print('finished openshoe:', float(index) /
                      self.ImuSourceData.shape[0])
            if index > 1:
                ins_filter.para.Ts = self.ImuSourceData[index, 0] - \
                                     self.ImuSourceData[index - 1, 0]
            self.openshoeresult[index, 0] = self.ImuSourceData[index, 0]
            self.openshoeresult[index, 1:] = ins_filter.GetPosition(
                self.ImuSourceData[index, 1:7],
                self.ImuSourceData[index, 1:7],
                zupt1[index],
                zupt1[index]).reshape([18])
        '''
        Test openshoe result.
        '''
        plt.figure(1)
        plt.plot(self.openshoeresult[:, 1], self.openshoeresult[:, 2],
                 'r+-')
        plt.grid(True)

        # plt.show()

    def UWBRun(self):
        '''

        :return:
        '''

        '''
        Synchronize
        '''

        self.ImuResultSyn = np.zeros([self.UwbData.shape[0], 3])

        index = 0
        for i in range(self.UwbData.shape[0]):
            uwb_time = self.UwbData[i, 0]
            # print(uwb_time)
            while (np.abs(uwb_time - self.openshoeresult[index, 0]) > 0.1):
                # print(uwb_time,self.openshoeresult[index,0])
                index += 1
                if (index == self.openshoeresult.shape[0]):
                    index -= 1
                    print("Unexpected to run to much times.")
                    break
            self.ImuResultSyn[i, :] = self.openshoeresult[index, 1:4]
        print('ImuResultSyn shape:', self.ImuResultSyn.shape)

        plt.figure(11)
        plt.plot(self.ImuResultSyn[:, 0], self.ImuResultSyn[:, 1], 'b-+')

        plt.show()


if __name__ == '__main__':
    import os

    for dir_name in os.listdir('./'):
        print(dir_name)
        if '07-0' in dir_name:
            dc = DataChronic(dir_name)
            dc.RunOpenshoe()
            dc.UWBRun()
