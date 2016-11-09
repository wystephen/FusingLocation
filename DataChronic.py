# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 09　20:57

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from OPENSHOE import PdrEkf,Setting,zupt_test

import XimuDataPreProcess

from log_process import seq_process


class DataChronic:
    def __init__(self,dir_name):
        xdpp = XimuDataPreProcess.XimuDataPreProcess(dir_name)

        self.ImuSourceData = xdpp.data_index

        tmp_data  = self.ImuSourceData.copy()

        self.ImuSourceData[:,1:4] = tmp_data[:,4:7] * 9.8
        self.ImuSourceData[:,4:7] = tmp_data[:,1:4] * np.pi / 180.0

        se = seq_process()

        import  os
        for file_name in os.listdir(dir_name):
            if '.data' in file_name:
                # print("file__name :", file_name)
                se.process_file(file_name=dir_name + '/' + file_name)

        self.UwbData = np.loadtxt('atrange.txt')

        print(self.UwbData.shape)
        print(self.ImuSourceData.shape)
    def RunOpenshoe(self):
        self.openshoeresult = np.zeros([self.UwbData.shape[0],3])

        '''
        Run Openshoe output the result.
        '''



if __name__ == '__main__':
    dc = DataChronic('10-03-04-00-01')





