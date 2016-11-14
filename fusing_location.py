# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 14　10:31

import os
import sys
import math
import time,timeit

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


from XimuDataPreProcess import XimuDataPreProcess
from DataChronic import DataChronic
from ViewerModel import PF_FRAME

from TranglePose import tranglepose

class fusing_location:
    def __init__(self,dir_name):
        #---
        self.dc = DataChronic(dir_name)
        self. dc.RunOpenshoe()
        self. dc.SynData()
        # self. dc.OnlyPF(100)

        # Copy data to this class
        self.BeaconSet = self.dc.BeaconSet
        self.UwbData = self.dc.UwbData
        self.ImuResultSyn = self.dc.ImuResultSyn


        '''
        Need to compute:
        1. Z_OFFSET and INITIAL POINT
        2.
        '''
        tp  = tranglepose(self.BeaconSet,self.UwbData[0:10,1:])





    def OnlyPF(self, particle_num=200):
        '''

        :param particle_num:
        :return:
        '''

        '''
        PF ONLY USE UWB DATA
        '''
        print(self.BeaconSet)

        self.pf = PF_FRAME.PF_Frame([1000, 1000], [10, 10], 10, particle_num)

        self.pf.SetBeaconSet(self.BeaconSet[:, 0:2])
        self.pf.InitialPose([0.0, 0.0])

        self.UWBResult = np.zeros([self.UwbData.shape[0], 2])
        # print(self.UwbData,self.UwbData.shape)

        self.UwbData /= 1000.0
        #
        # self.UwbData = self.UwbData ** 2.0
        # self.UwbData -= self.z_offset
        # self.UwbData = self.UwbData ** 0.5

        self.UwbData[:, 1:] = (self.UwbData[:, 1:] ** 2.0 - self.z_offset)

        self.UwbData[:, 1:] = np.sqrt(np.abs(self.UwbData[:, 1:]))

        plt.figure(111104)
        for i in range(self.UwbData.shape[1]):
            if i > 0:
                plt.plot(self.UwbData[:, i])
        # plt.show()

        for i in range(self.UwbData.shape[0]):
            self.pf.Sample(0.3)
            self.pf.Evaluated(self.UwbData[i, 1:5])

            self.pf.ReSample()

            self.UWBResult[i, :] = self.pf.GetResult()








if __name__ == '__main__':
    for dir_name in os.listdir('./'):
        if '05-0' in dir_name:
            location = fusing_location(dir_name)