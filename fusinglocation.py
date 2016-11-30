# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 14　10:31

import os
import sys
import math
import time, timeit

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from XimuDataPreProcess import XimuDataPreProcess
from DataChronic import DataChronic
from ViewerModel import PF_FRAME

from TranglePose import trianglepose


class FusingLocation:
    def __init__(self, dir_name, beacon_use=[0, 1, 2, 3]):
        # ---
        self.dc = DataChronic(dir_name, beacon_use=beacon_use)
        self.dc.RunOpenshoe()
        self.dc.SmoothPath()
        self.dc.SynData()
        # self. dc.OnlyPF(100)

        # Copy data to this class
        self.BeaconSet = self.dc.BeaconSet
        self.UwbData = self.dc.UwbData
        self.ImuResultSyn = self.dc.ImuResultSyn

        self.UwbData[:, 1:] = self.UwbData[:, 1:] / 1000.0

        '''
        Need to compute:
        1. Z_OFFSET and INITIAL POINT
        2.
        '''
        tp = trianglepose(self.BeaconSet, self.UwbData[2:10, 1:])
        self.z_offset = tp.pose[2] - self.BeaconSet[1, 2]
        self.initialpose = tp.pose[0:2]
        self.OptResult = tp.ComputePath(self.UwbData)
        self.TriResult = tp.TriComputePath(self.UwbData)

        '''
        Save data to file
        '''
        tmp_dir_name = './tmp_file_dir/'

        np.savetxt(tmp_dir_name + 'beaconset.data', self.BeaconSet)
        np.savetxt(tmp_dir_name + 'UwbData.data', self.UwbData)
        np.savetxt(tmp_dir_name + 'UwbResult.data', self.OptResult)

        np.savetxt(tmp_dir_name + 'ImuData.data', self.dc.ImuSourceData)
        np.savetxt(tmp_dir_name + 'Zupt.data', self.dc.zupt)
        np.savetxt(tmp_dir_name + 'ImuResultData.data', self.ImuResultSyn)



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
        self.pf.InitialPose(self.initialpose)
        # self.pf.InitialPose([0.0, 0.0])

        self.UWBResult = np.zeros([self.UwbData.shape[0], 2])
        # print(self.UwbData,self.UwbData.shape)

        # self.UwbData /= 1000.0
        #
        # self.UwbData = self.UwbData ** 2.0
        # self.UwbData -= self.z_offset
        # self.UwbData = self.UwbData ** 0.5

        self.UwbData[:, 1:] = np.abs(self.UwbData[:, 1:] ** 2.0 - self.z_offset)

        self.UwbData[:, 1:] = np.sqrt(np.abs(self.UwbData[:, 1:]))

        plt.figure(111104)
        for i in range(self.UwbData.shape[1]):
            if i > 0:
                plt.plot(self.UwbData[:, i])
        # plt.show()

        for i in range(self.UwbData.shape[0]):
            self.pf.Sample(0.5)
            self.pf.Evaluated(self.UwbData[i, 1:5])

            self.pf.ReSample()

            self.UWBResult[i, :] = self.pf.GetResult()
        plt.figure(1)
        plt.title("uwb-r:opt-y:tri-g")
        plt.plot(self.UWBResult[:, 0], self.UWBResult[:, 1], 'r-+')
        plt.plot(self.OptResult[:, 1], self.OptResult[:, 2], 'y-+')
        if self.TriResult.shape[0] == self.OptResult.shape[0]:
            plt.plot(self.TriResult[:, 1], self.TriResult[:, 2], 'g-+')

        plt.grid(True)
        # plt.show()

    def Transform(self):
        from reference_transform import ReferenceTransform
        self.tf = ReferenceTransform()
        # print(self.ImuResultSyn.shape)
        self.tf.SetOffset(self.UWBResult[0, 0:2])
        self.tf.EstimateTheta(self.ImuResultSyn, self.UWBResult)
        self.ImuSynT = self.tf.tmp_imu_path

    def MixFusing(self, particle_num=200):
        self.pf = PF_FRAME.PF_Frame([1000, 1000], [10, 10], 10, particle_num)

        self.pf.SetBeaconSet(self.BeaconSet[:, 0:2])
        self.pf.InitialPose(self.initialpose)
        # self.pf.InitialPose([0.0, 0.0])

        self.FusingResult = np.zeros([self.UwbData.shape[0], 2])
        # print(self.UwbData,self.UwbData.shape)

        # self.UwbData /= 1000.0
        #
        # self.UwbData = self.UwbData ** 2.0
        # self.UwbData -= self.z_offset
        # self.UwbData = self.UwbData ** 0.5

        self.UwbData[:, 1:] = np.abs(self.UwbData[:, 1:] ** 2.0 - self.z_offset)

        self.UwbData[:, 1:] = np.sqrt(np.abs(self.UwbData[:, 1:]))

        # plt.figure(111104)
        # for i in range(self.UwbData.shape[1]):
        #     if i > 0:
        #         plt.plot(self.UwbData[:, i])
        # plt.show()

        for i in range(self.UwbData.shape[0]):
            # self.pf.Sample(0.5)
            if 18 > i > 2:
                '''
                odometry method 1
                '''
                self.pf.OdometrySample(self.ImuSynT[i, :] - self.ImuSynT[i - 1, :],
                                       0.2)
            elif i > 18:
                '''
                Odometry method 2
                '''
                # vec_last = self.ImuResultSyn[i - 1, 1:] - self.ImuResultSyn[i - 6, 1:]  # last time odo
                vec_now = self.ImuSynT[i, :] - self.ImuSynT[i - 1, :]  # this time odo

                # vec_res = self.FusingResult[i - 1, :] - self.FusingResult[i - 6, :]  # last time result

                odo_vec = self.tf.ComputeRefOdo(vec_now,
                                                # self.UWBResult[i - 17:i - 1, :],
                                                self.FusingResult[i - 17:i - 1, :],
                                                self.ImuSynT[i - 17:i - 1, :])

                # self.tmp_imu = self.tf.Transform(self.ImuResultSyn[:,1:3])
                print("para:", vec_now, odo_vec)

                # self.pf.OdometrySample(self.tmp_imu[i, :] - self.tmp_imu[i - 1, :], 0.1)
                self.pf.OdometrySample(odo_vec, 0.1)

            else:
                self.pf.Sample(0.5)

            self.pf.Evaluated(self.UwbData[i, 1:5], sigma=3.0)

            self.pf.ReSample()

            self.FusingResult[i, :] = self.pf.GetResult()

        plt.figure(119)
        plt.title("MIX FUSING b-imu r-fusing g-uwb")

        # plt.plot(self.tmp_imu[:, 0], self.tmp_imu[:, 1], 'b-+')

        plt.plot(self.ImuSynT[:, 0], self.ImuSynT[:, 1], 'b-+')

        plt.plot(self.FusingResult[:, 0], self.FusingResult[:, 1], 'r-+')
        plt.plot(self.UWBResult[:, 0], self.UWBResult[:, 1], 'g-+')
        plt.grid(True)

    def Fusing(self, particle_num=200):
        self.pf = PF_FRAME.PF_Frame([1000, 1000], [10, 10], 10, particle_num)

        self.pf.SetBeaconSet(self.BeaconSet[:, 0:2])
        self.pf.InitialPose(self.initialpose)
        # self.pf.InitialPose([0.0, 0.0])

        self.FusingResult = np.zeros([self.UwbData.shape[0], 2])
        # print(self.UwbData,self.UwbData.shape)

        # self.UwbData /= 1000.0
        #
        # self.UwbData = self.UwbData ** 2.0
        # self.UwbData -= self.z_offset
        # self.UwbData = self.UwbData ** 0.5

        self.UwbData[:, 1:] = np.abs(self.UwbData[:, 1:] ** 2.0 - self.z_offset)

        self.UwbData[:, 1:] = np.sqrt(np.abs(self.UwbData[:, 1:]))


        for i in range(self.UwbData.shape[0]):
            # self.pf.Sample(0.5)
            if 18 > i > 2 or True:
                '''
                odometry method 1
                '''
                self.pf.OdometrySample(self.ImuSynT[i, :] - self.ImuSynT[i - 1, :],
                                       0.1)
            elif i > 18:
                '''
                Odometry method 2
                '''
                # vec_last = self.ImuResultSyn[i - 1, 1:] - self.ImuResultSyn[i - 6, 1:]  # last time odo
                vec_now = self.ImuResultSyn[i, 1:] - self.ImuResultSyn[i - 1, 1:]  # this time odo

                # vec_res = self.FusingResult[i - 1, :] - self.FusingResult[i - 6, :]  # last time result

                odo_vec = self.tf.ComputeRefOdo(vec_now,
                                                self.FusingResult[i - 6:i - 1, :],
                                                self.ImuResultSyn[i - 6:i - 1, 1:])

                self.pf.OdometrySample(odo_vec, 0.06)

            else:
                self.pf.Sample(0.5)

            self.pf.Evaluated(self.UwbData[i, 1:5], sigma=3.0)

            self.pf.ReSample()

            self.FusingResult[i, :] = self.pf.GetResult()

        plt.figure(2211)
        plt.title("fusing v.s. pf")
        plt.plot(self.UWBResult[:, 0], self.UWBResult[:, 1], 'r-+')
        plt.plot(self.FusingResult[:, 0], self.FusingResult[:, 1], 'b-+')
        plt.grid(True)

    def DeepFusing(self, particle_num):

        self.pf = PF_FRAME.PF_Frame([1000, 1000], [10, 10], 10, particle_num)

        self.pf.SetBeaconSet(self.BeaconSet[:, 0:2])
        self.pf.InitialPose(self.initialpose)

        self.FusingResult = np.zeros([self.UwbData.shape[0], 2])

        self.UwbData[:, 1:] = np.abs(self.UwbData[:, 1:] ** 2.0 - self.z_offset)

        self.UwbData[:, 1:] = np.sqrt(np.abs(self.UwbData[:, 1:]))

        '''
        Openshoe prepare.
        '''

        self.ImuData = self.dc.ImuSourceData

        # tmp = self.ImuData.copy()
        #
        # self.ImuData[:,1] = tmp[:,2]
        # self.ImuData[:,2] = tmp[:,1]
        #
        # self.ImuData[:,4] = tmp[:,5]
        # self.ImuData[:,5] = tmp[:,4]



        from OPENSHOE.Setting import settings

        para = settings()

        from reference_transform import ReferenceTransform

        self.tf = ReferenceTransform()

        self.tf.SetOffset(self.UWBResult[0, 0:2])

        self.tf.EstimateTheta(self.ImuResultSyn, self.UWBResult)

        para.init_heading1 = para.init_heading2 = -self.tf.theta
        # print("theta is :" , self.tf.theta / np.pi * 180.0)
        para.init_pos1 = para.init_pos2 = np.asarray(
            [self.initialpose[0], self.initialpose[1], 0.0]
        )

        print("para heading and pos:", para.init_heading1, para.init_pos1)

        para.Ts = np.mean(self.ImuData[1:, 0] - self.ImuData[0:-1, 0])
        para.min_rud_sep = int(1 / para.Ts)

        para.time_Window_size = 5
        para.gamma = 6580

        para.range_constraint_on = False

        self.zupt = self.dc.zupt

        from OPENSHOE.PdrEkfPlus import ZUPTaidedInsPlus

        self.ins = ZUPTaidedInsPlus(para)

        self.ins.init_Nav_eq(self.ImuData[1:50, 1:7],
                             self.ImuData[1:50, 1:7])

        print("imu data :", self.ImuData.shape)
        print("Uwb data;", self.UwbData.shape)

        print(np.mean(self.ImuData[:, 0]), np.mean(self.UwbData[:, 0]))

        print("-------------")

        '''

        '''

        imu_index = 0
        uwb_index = 0

        self.ImuResultFusing = np.zeros([self.ImuData.shape[0], 19])
        self.UwbResultFusing = np.zeros([self.UwbData.shape[0], 3])

        LastImuPose = self.initialpose

        while True:
            if imu_index == self.ImuData.shape[0] or \
                            uwb_index == self.UwbData.shape[0]:
                break

            if uwb_index == 0:
                self.pf.Sample(0.5)

                self.pf.Evaluated(self.UwbData[uwb_index, 1:], 1.0)
                self.pf.ReSample()

                self.UwbResultFusing[uwb_index, 0] = self.UwbData[uwb_index, 0]
                self.UwbResultFusing[uwb_index, 1:] = self.pf.GetResult()

                uwb_index += 1
                continue

            if imu_index == 0:
                self.ImuResultFusing[imu_index, 0] = self.ImuData[imu_index, 0]
                self.ImuResultFusing[imu_index, 1:] = self.ins.GetPosition(
                    self.ImuData[imu_index, 1:7],
                    self.zupt[imu_index],
                    self.UwbResultFusing[uwb_index, 1:],
                    3.0
                ).reshape([18])

                # tmp = self.ImuResultFusing[imu_index, :]
                # self.ImuResultFusing[imu_index, 1] = tmp[2]
                # self.ImuResultFusing[imu_index, 2] = tmp[1]

                imu_index += 1
                continue

            if self.UwbData[uwb_index, 0] < self.ImuData[imu_index, 0]:
                odo_vec = self.ImuResultFusing[imu_index - 1, 1:3] - LastImuPose

                self.pf.OdometrySample(odo_vec, 0.1)

                self.pf.Evaluated(self.UwbData[uwb_index, 1:], sigma=3.0)

                self.pf.ReSample()

                self.UwbResultFusing[uwb_index, 0] = self.UwbData[uwb_index, 0]
                self.UwbResultFusing[uwb_index, 1:] = self.pf.GetResult()

                LastImuPose = self.ImuResultFusing[imu_index - 1, 1:3]

                uwb_index += 1

                continue
            else:

                self.ImuResultFusing[imu_index, 0] = self.ImuData[imu_index, 0]

                range_constraint_value = 5.4
                if imu_index < 10:
                    range_constraint_value = 1000.0
                elif imu_index < 20:
                    range_constraint_value = 6000.0
                else:
                    range_constraint_value = 03000.0

                self.ImuResultFusing[imu_index, 1:] = self.ins.GetPosition(
                    self.ImuData[imu_index, 1:7],
                    self.zupt[imu_index],
                    self.UwbResultFusing[uwb_index, 1:],
                    range_constraint_value
                ).reshape([18])

                # tmp = self.ImuResultFusing[imu_index, :]
                # self.ImuResultFusing[imu_index, 1] = tmp[2]
                # self.ImuResultFusing[imu_index, 2] = tmp[1]

                imu_index += 1

                continue

        plt.figure(911)
        plt.title("DEEP FUSING")

        plt.plot(self.ImuResultFusing[:, 1], self.ImuResultFusing[:, 2], 'r-+')
        plt.plot(self.UwbResultFusing[:, 1], self.UwbResultFusing[:, 2], 'b-+')
        plt.plot(self.UWBResult[:, 0], self.UWBResult[:, 1], 'y-+')

        plt.grid(True)


if __name__ == '__main__':
    ex_dir_list = list()
    for dir_name in os.listdir('./'):
        if '-0' in dir_name and not '0.' in dir_name:
            ex_dir_list.append(dir_name)
            # if '03-' in dir_name:  # or '-0'in dir_name:
            #     print(dir_name)
            #     location = FusingLocation(dir_name)
            #     location.OnlyPF()
            #     location.Transform()
            #     location.Fusing(200)
            #
            #     plt.show()
    for i in [3]:
        dir_name = ex_dir_list[i]
        print(dir_name)
        location = FusingLocation(dir_name, [0, 1, 2, 3])
        location.OnlyPF()
        location.Transform()
        # location.Fusing(1000)
        # location.DeepFusing(1000)
        location.MixFusing(1000)

        plt.show()
