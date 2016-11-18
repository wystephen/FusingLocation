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
    def __init__(self, dir_name):
        # ---
        self.dc = DataChronic(dir_name)
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
        plt.plot(self.UWBResult[:, 0], self.UWBResult[:, 1], 'r-+')
        plt.plot(self.OptResult[:, 1], self.OptResult[:, 2], 'y-+')
        plt.grid(True)
        # plt.show()

    def Transform(self):
        from reference_transform import ReferenceTransform
        tf = ReferenceTransform()
        # print(self.ImuResultSyn.shape)
        tf.SetOffset(self.UWBResult[0, 0:2])
        tf.EstimateTheta(self.ImuResultSyn, self.UWBResult)
        self.ImuSynT = tf.tmp_imu_path

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

        # plt.figure(111104)
        # for i in range(self.UwbData.shape[1]):
        #     if i > 0:
        #         plt.plot(self.UwbData[:, i])
        # plt.show()

        for i in range(self.UwbData.shape[0]):
            # self.pf.Sample(0.5)
            if 8 > i > 2 or True:
                '''
                odometry method 1
                '''
                self.pf.OdometrySample(self.ImuSynT[i, :] - self.ImuSynT[i - 1, :], 0.2)
            elif i > 8:
                '''
                Odometry method 2
                '''
                vec_last = self.ImuResultSyn[i - 1, 1:] - self.ImuResultSyn[i - 5, 1:]  # last time odo
                vec_now = self.ImuResultSyn[i, 1:] - self.ImuResultSyn[i - 1, 1:]  # this time odo

                vec_res = self.FusingResult[i - 1, :] - self.FusingResult[i - 3, :]  # last time result

                # print('vec shape',vec_last.shape,vec_now.shape,vec_res.shape)
                #
                # theta_offset = vec_last.dot(vec_now) / \
                #                np.linalg.norm(vec_last) / \
                #                np.linalg.norm(vec_now)
                theta_offset = math.atan2(vec_now[1], vec_now[0]) - \
                               math.atan2(vec_last[1], vec_last[0])

                theta_src = math.atan2(vec_res[1], vec_res[0])

                theta = theta_src + theta_offset
                #
                # print("theta",math.atan2(vec_now[1], vec_now[0]),
                #       math.atan2(vec_last[1], vec_last[0]),
                #       theta_src)


                odo_vec = np.asarray([np.linalg.norm(vec_now) * np.cos(theta),
                                      np.linalg.norm(vec_now) * np.sin(theta)],
                                     dtype=float)

                # odo_vec = vec_now / vec_last * vec_res
                #
                # vec_last_n = vec_last / np.linalg.norm(vec_last)
                # vec_now_n = vec_now / np.linalg.norm(vec_now)
                #
                # vec_res_n = vec_res / np.linalg.norm(vec_res)

                # odo_vec = vec_res + vec_now - vec_last
                # odo_vec /= np.linalg.norm(odo_vec)

                self.pf.OdometrySample(odo_vec, 0.1)

            else:
                self.pf.Sample(0.5)

            self.pf.Evaluated(self.UwbData[i, 1:5])

            self.pf.ReSample()

            self.FusingResult[i, :] = self.pf.GetResult()

        plt.figure(2211)
        plt.title("fusing v.s. pf")
        plt.plot(self.UWBResult[:, 0], self.UWBResult[:, 1], 'r-+')
        plt.plot(self.FusingResult[:, 0], self.FusingResult[:, 1], 'b-+')
        plt.grid(True)


if __name__ == '__main__':
    for dir_name in os.listdir('./'):
        if '03-02' in dir_name:  # or '-0'in dir_name:
            print(dir_name)
            location = FusingLocation(dir_name)
            location.OnlyPF()
            location.Transform()
            location.Fusing(200)

            plt.show()
