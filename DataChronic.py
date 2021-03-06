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
    def __init__(self, dir_name, beacon_use=[0, 1, 2, 3]):
        '''
        Load and preprocess data.

        :param dir_name:
        '''

        '''
        OFFSET
        '''

        self.z_offset = 0.65
        self.time_offset = 0.0

        self.time_offset = 531844066.535

        self.time_offset += 1.10

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
            if '.data' in file_name and 'LOG' in file_name:
                print("file__name :", file_name)
                se.process_file(file_name=dir_name + '/' + file_name)

        self.UwbData = np.loadtxt('atrange.txt')
        self.BeaconSet = np.loadtxt(dir_name + '/' + 'beaconset')

        print("UWB HZ AVERAGE TS IS :",
              np.mean(self.UwbData[1:, 0] - self.UwbData[:-1, 0]))

        o_beacon_use = np.zeros(np.asarray(beacon_use).shape[0] + 1)
        o_beacon_use[1:] = beacon_use
        o_beacon_use[1:] = o_beacon_use[1:] + 1
        self.UwbData = self.UwbData[:, o_beacon_use.astype(dtype=int)]
        self.BeaconSet = self.BeaconSet[beacon_use, :]

        '''
        Single value Kalman filter for uwb range
        '''
        self.estcov = np.zeros_like(self.UwbData[:, 1:])
        self.meacov = np.zeros_like(self.UwbData[:, 1:])
        for i in range(self.UwbData.shape[1]):
            if i == 0:
                continue

            EstimateCov = 0.5
            MeasureCov = 0.5

            Estimate = 0.0

            e_cov_list = list()
            m_cov_list = list()

            # for j in range(self.UwbData.shape[0]):
            #     K = EstimateCov * np.sqrt(1 / (EstimateCov * EstimateCov + MeasureCov * MeasureCov))
            #     Estimate = Estimate + K * (self.UwbData[j, i] - Estimate)
            #
            #     EstimateCov = np.sqrt(1 - K) * EstimateCov
            #
            #     MeasureCov = np.sqrt(1 - K) * MeasureCov
            #
            #     e_cov_list.append(EstimateCov)
            #     m_cov_list.append(MeasureCov)
            #
            #     self.UwbData[j, i] = Estimate
            # self.estcov[:, i - 1] = np.asarray(e_cov_list)
            # self.meacov[:, i - 1] = np.asarray(m_cov_list)

        # plt.figure(9911)
        # plt.title("cov of estimate")
        # plt.grid(True)

        # for i in range(self.estcov.shape[1]):
        #     plt.plot(self.estcov[:, i])
        #
        # plt.figure(9912)
        # plt.title("cov of measure")
        # plt.grid(True)
        #
        # for i in range(self.meacov.shape[1]):
        #     plt.plot(self.meacov[:, i])

        '''
        Test change BeaconSet
        '''
        # self.BeaconSet[:,2] *= 1.0
        tmp = self.BeaconSet.copy()

        self.BeaconSet[:, 0], self.BeaconSet[:, 1] = tmp[:, 0], tmp[:, 1]*-1.0

        '''
        Add Time offset
        '''
        if np.abs(np.mean(self.ImuSourceData[:, 0]) -
                          np.mean(self.UwbData[:, 0])) > 1000.0:
            self.ImuSourceData[:, 0] += self.time_offset
            print("-------ADDET")

            # d_index = [32,60,75,96,110,135,147,172,205,232,244,268,282,303,315,344,379]
            #
            # key_point_time = list()
            #
            # for i in d_index:
            #     key_point_time.append(self.UwbData[i, 0])
            # key_point_time = np.asarray(key_point_time)

            # key_point_tmp = np.loadtxt(dir_name + '/' + 'keypointtmp.t')
            # print(key_point_time.shape,"     key point    ",key_point_tmp.shape)
            #
            # key_point_data = np.zeros([key_point_tmp.shape[0],key_point_tmp.shape[1]+1])
            # for i in range(key_point_tmp.shape[0]):
            #     key_point_data[i,0:2] = key_point_tmp[i,:]
            #     key_point_data[i,2] = key_point_time[i]
            # np.savetxt(dir_name+'/keypoint.csv',key_point_data,delimiter=',')

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
        # print("Ts:", setting.Ts)

        setting.min_rud_sep = int(1 / setting.Ts)

        setting.time_Window_size = 5
        setting.gamma = 6580

        setting.init_heading2 = setting.init_heading1
        setting.init_heading1 = setting.init_heading2 = 0.0

        zupt_detector = zupt_test.zupte_test(setting)

        zupt1 = zupt_detector.GLRT_Detector(self.ImuSourceData[:, 1:7])
        self.zupt = zupt1

        plt.figure(1110112)
        plt.plot(self.ImuSourceData[:, 0], zupt1 * 12000, 'r+-')
        plt.plot(self.UwbData[:, 0], self.UwbData[:, 1], 'g+')

        # print("MARK1",self.ImuSourceData[:,0])

        ins_filter = PdrEkf.ZUPTaidedIns(setting)

        ins_filter.init_Nav_eq(self.ImuSourceData[1:40, 1:7],
                               self.ImuSourceData[1:40, 1:7])

        for index in range(self.ImuSourceData.shape[0]):
            # if (index % 5000 == 0):
            #     print('finished openshoe:', float(index) /
            #           self.ImuSourceData.shape[0])
            if index > 1:
                ins_filter.para.Ts = self.ImuSourceData[index, 0] - \
                                     self.ImuSourceData[index - 1, 0]
            self.openshoeresult[index, 0] = self.ImuSourceData[index, 0]
            self.openshoeresult[index, 1:] = ins_filter.GetPosition(
                self.ImuSourceData[index, 1:7],
                self.ImuSourceData[index, 1:7],
                zupt1[index],
                zupt1[index]).reshape([18])
        plt.plot(self.ImuSourceData[:, 0], self.openshoeresult[:, 1] * 1000, 'b-+')
        plt.plot(self.ImuSourceData[:, 0], self.openshoeresult[:, 2] * 1000, 'y-+')
        '''
        Test openshoe result.
        '''
        plt.figure(1111102)
        plt.plot(self.openshoeresult[:, 1], self.openshoeresult[:, 2],
                 'r+-')
        plt.grid(True)

        # plt.show()

    def SmoothPath(self):
        # self.openshoeresult *= 1.0
        # print('aa')
        # print(self.zupt)

        '''
        Compute length of every step
        '''
        # Ignore several step near the start point.
        ignstep = 4
        steplen = 0
        lenvec = list()

        for i in range(self.zupt.shape[0] - 1):
            if self.zupt[i] == 1 and self.zupt[i + 1] == 0:
                ignstep -= 1
                if ignstep < 0:
                    lenvec.append(steplen)

            elif self.zupt[i] == 0 and self.zupt[i + 1] == 1:
                steplen = 0
            else:
                steplen += 1
        # print("average len:", np.mean(np.asarray(lenvec)))

        '''
        Add offset and generate a new path.
        '''
        t_offset = int(np.mean(np.asarray(lenvec)))
        tmp_openshoe_src = np.zeros([self.openshoeresult.shape[0] + t_offset,
                                     self.openshoeresult.shape[1]])
        tmp_openshoe_src[t_offset:, :] = self.openshoeresult.copy()
        tmp_openshoe_src[:t_offset, :] = self.openshoeresult[t_offset:2 * t_offset, :]
        tmp_openshoe_ori = self.openshoeresult.copy()

        for i in range(self.openshoeresult.shape[0]):
            self.openshoeresult[i, 1:] = (tmp_openshoe_src[i, 1:]
                                          + tmp_openshoe_ori[i, 1:]) / 2.0

            # plt.plot(self.openshoeresult[:, 1], self.openshoeresult[:, 2], 'y-+')
            # plt.figure(1102023)
            # plt.plot(self.openshoeresult[:, 0], self.openshoeresult[:, 1], 'r-+')
            # plt.plot(self.openshoeresult[:, 0], self.openshoeresult[:, 2], 'b-+')
            # plt.grid(True)

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
            uwb_time = self.UwbData[i, 0]

            for j in range(self.openshoeresult.shape[0]):
                if np.abs(uwb_time - self.openshoeresult[j, 0]) < 0.05:
                    self.ImuResultSyn[i, :] = self.openshoeresult[j, 1:4]
                    break
                if j == self.openshoeresult.shape[0] - 1:
                    print("MAYBE SOME ERROR HERE")
                    # self.ImuResultSyn = self.ImuResultSyn[:i,:]
                    # i = self.UwbData.shape[0]
                    break
                    # if i == self.UwbData.shape[0] and i:
                    #     break


        '''
        IMPORTANT MODIFICATE
        '''
        # tmp = self.ImuResultSyn.copy()
        # self.ImuResultSyn[:, 0] = tmp[:, 1]
        # self.ImuResultSyn[:, 1] = tmp[:, 0]

        plt.figure(111101)
        plt.plot(self.ImuResultSyn[:, 0], self.ImuResultSyn[:, 1], 'b-+')


        # plt.figure(1010101)
        # plt.title('IMU sperate')
        # plt.plot(self.ImuResultSyn[:,0],'r-+')
        # plt.plot(self.ImuResultSyn[:,1],'b-+')
        # plt.grid(True)
        #
        # d_index = [32,60,75,96,110,135,147,172,205,232,244,268,282,303,315,344,379]
        # plt.figure(1010102)
        # plt.title('Show Key Point')
        # plt.plot(self.ImuResultSyn[d_index,0],self.ImuResultSyn[d_index,1],'r-+')
        # plt.grid(True)

        # plt.show()

    def OnlyPF(self, particle_num=200):
        '''

        :param particle_num:
        :return:
        '''

        '''
        PF ONLY USE UWB DATA
        '''
        # print(self.BeaconSet)

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

        plt.figure(11114)
        for i in range(self.UwbData.shape[1]):
            if i > 0:
                plt.plot(self.UwbData[:, i])

        # plt.show()
        plt.figure(15)
        for j in range(self.UwbData.shape[1]):
            if j > 0:
                plt.plot(self.UwbData[1:, j] - self.UwbData[0:-1, j])
                print("run hear", j, i)
        plt.grid(True)

        for i in range(self.UwbData.shape[0]):
            self.pf.Sample(0.5)
            self.pf.Evaluated(self.UwbData[i, 1:5])

            self.pf.ReSample()
            self.UWBResult[i, :] = self.pf.GetResult()

        plt.figure(22)
        plt.plot(self.UWBResult[:, 0], self.UWBResult[:, 1], 'g+-')
        plt.grid(True)


if __name__ == '__main__':
    import os

    for dir_name in os.listdir('./'):
        # print(dir_name)
        if '04-02-' in dir_name:
            print(dir_name)
            dc = DataChronic(dir_name)
            dc.RunOpenshoe()
            dc.SynData()
            dc.OnlyPF(200)

            plt.show()
