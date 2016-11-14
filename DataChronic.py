# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 09　20:57

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from OPENSHOE import PdrEkf, Setting, zupt_test

import XimuDataPreProcess

from log_process import seq_process

from ViewerModel import PF_FRAME


class DataChronic:
    def __init__(self, dir_name):
        '''
        Load and preprocess data.

        :param dir_name:
        '''

        '''
        OFFSET
        '''

        self.z_offset = 0.65

        self.time_offset =  531844066.535

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
        Add Time offset
        '''
        self.ImuSourceData[:,0] += self.time_offset


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
        plt.figure(1111102)
        plt.plot(self.openshoeresult[:, 1], self.openshoeresult[:, 2],
                 'r+-')
        plt.grid(True)

        # plt.show()

    def SynData(self):
        '''

        :return:
        '''

        '''
        Synchronize
        '''

        self.ImuResultSyn = np.zeros([self.UwbData.shape[0], 3])

        index = 0
        # for i in range(self.UwbData.shape[0]):
        #     uwb_time = self.UwbData[i, 0]
        #     # print(uwb_time)
        #     while (np.abs(uwb_time - self.openshoeresult[index, 0]) > 0.1):
        #         print(uwb_time,self.openshoeresult[index,0])
        #         index += 1
        #         if (index == self.openshoeresult.shape[0]):
        #             index -= 1
        #             # print("Unexpected to run to much times.")
        #             break
        #     self.ImuResultSyn[i, :] = self.openshoeresult[index, 1:4]
        # print('ImuResultSyn shape:', self.ImuResultSyn.shape)

        for i in range(self.UwbData.shape[0]):
            uwb_time = self.UwbData[i,0]

            for j in range(index-5,self.openshoeresult.shape[0]):
                if np.abs(uwb_time-self.openshoeresult[j,0])<0.05:
                    self.ImuResultSyn[i,:] = self.openshoeresult[j,1:4]
                    break
                if j == self.openshoeresult.shape[0]-1:
                    print("MAYBE SOME ERROR HERE")

        plt.figure(111101)
        plt.plot(self.ImuResultSyn[:, 0], self.ImuResultSyn[:, 1], 'b-+')

        # plt.show()

    def OnlyPF(self,particle_num = 200):
        '''

        :param particle_num:
        :return:
        '''

        '''
        PF ONLY USE UWB DATA
        '''
        print(self.BeaconSet)

        self.pf = PF_FRAME.PF_Frame([1000,1000],[10,10],10,particle_num)

        self.pf.SetBeaconSet(self.BeaconSet[:,0:2])
        self.pf.InitialPose([0.0,0.0])

        self.UWBResult = np.zeros([self.UwbData.shape[0],2])
        # print(self.UwbData,self.UwbData.shape)

        self.UwbData /= 1000.0
        #
        # self.UwbData = self.UwbData ** 2.0
        # self.UwbData -= self.z_offset
        # self.UwbData = self.UwbData ** 0.5

        self.UwbData[:,1:] = (self.UwbData[:,1:] ** 2.0 - self.z_offset)

        self.UwbData[:,1:] = np.sqrt(np.abs(self.UwbData[:,1:]))



        plt.figure(111104)
        for i in range(self.UwbData.shape[1]):
            if i >0:
                plt.plot(self.UwbData[:, i])
        # plt.show()

        for i in range(self.UwbData.shape[0]):
            self.pf.Sample(0.3)
            self.pf.Evaluated(self.UwbData[i, 1:5])

            self.pf.ReSample()


            self.UWBResult[i,:]  = self.pf.GetResult()

        plt.figure(22)
        plt.plot(self.UWBResult[:,0],self.UWBResult[:,1],'g+-')
        plt.grid(True)





if __name__ == '__main__':
    import os

    for dir_name in os.listdir('./'):
        # print(dir_name)
        if '05-0' in dir_name:
            print(dir_name)
            dc = DataChronic(dir_name)
            dc.RunOpenshoe()
            dc.SynData()
            dc.OnlyPF(200)


            plt.show()
