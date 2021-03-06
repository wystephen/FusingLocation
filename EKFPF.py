# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 20　21:00



import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import os

import time

import copy

class EkfPf:
    def __init__(self,
                 beaconset,
                 UwbData,
                 UwbResult,
                 ImuData,
                 Zupt,
                 particle_number=1000):

        '''
        Test parameter here.
        '''

        particle_number = 150
        self.Sigma = 0.25
        self.EvaluateSigma = 1.0

        self.beaconset = beaconset
        self.UwbData = UwbData
        self.UwbResult = UwbResult

        self.ImuData = ImuData
        self.Zupt = Zupt

        self.particle_num = particle_number

        from OPENSHOE.Setting import settings

        self.main_para = settings()

        self.main_para.init_heading1 = self.main_para.init_heading2 = -np.pi  # -2.7035450517

        self.main_para.init_pos1 = self.main_para.init_pos2 = [0.8, -5.6, 0.0]

        self.main_para.range_constraint_on = False

        self.main_para.Ts = np.mean(self.ImuData[1:, 0] - self.ImuData[:-1, 0])
        self.main_para.min_rud_sep = int(1 / self.main_para.Ts)

        self.main_para.pose_constraint = False

        self.ekf_list = list()

        from SuperPdr import FusingPlus
        

        for i in range(self.particle_num):
            tmp_ins = FusingPlus(self.main_para)

            tmp_ins.init_Nav_eq(self.ImuData[1:20, 1:7],
                                self.ImuData[1:20, 1:7])

            self.ekf_list.append(tmp_ins)

        '''
        Achieve it as easy as possible.
        '''

        imu_index = 0
        uwb_index = 0

        self.FusingResult = np.zeros([self.UwbData.shape[0], 2])

        self.weight = np.ones(self.particle_num)
        self.weight /= np.sum(self.weight)

        self.poselist = np.zeros([self.particle_num, 2])

        time_last = time.time()

        while True:
            if imu_index == self.ImuData.shape[0] or uwb_index == self.UwbData.shape[0]:
                break

            if imu_index == 0:
                '''
                Sample
                '''
                for i in range(len(self.ekf_list)):
                    self.ekf_list[i].GetPosition(
                        self.ImuData[imu_index, 1:7] +
                        np.random.normal(0.0, self.Sigma, [6]),
                        self.Zupt[imu_index]
                    )
                imu_index += 1

            if uwb_index == 0:
                '''
                Initiao weight
                '''
                # print("beaconset",self.beaconset,self.beaconset.shape[0])
                for i in range(len(self.ekf_list)):
                    self.poselist[i, :], self.weight[i] = self.ekf_list[i].Evaluation(self.beaconset,
                                                                                      self.UwbData[uwb_index, 1:],
                                                                                      sigma=self.EvaluateSigma,
                                                                                      z_offset=1.9)
                self.weight /= np.sum(self.weight)

                tmp_pose = [0.0, 0.0]
                for i in range(self.weight.shape[0]):
                    tmp_pose += self.poselist[i, :] * self.weight[i]

                self.FusingResult[uwb_index, :] = tmp_pose

                uwb_index += 1

            if self.UwbData[uwb_index, 0] < self.ImuData[imu_index, 0]:
                '''
                Evaluation
                '''
                for i in range(len(self.ekf_list)):
                    self.poselist[i, :], score = \
                        self.ekf_list[i].Evaluation(self.beaconset,
                                                    self.UwbData[uwb_index, 1:],
                                                    sigma=self.EvaluateSigma,
                                                    z_offset=1.9
                                                    )
                    self.weight[i] *= score
                    # print("scoreL:", score)
                self.weight /= np.sum(self.weight)

                '''
                Result
                '''
                # self.FusingResult[uwb_index, :] = self.poselist * self.weight.reshape([self.particle_num])
                tmp_pose = [0.0, 0.0]
                for i in range(self.weight.shape[0]):
                    tmp_pose += self.poselist[i, :] * self.weight[i]

                self.FusingResult[uwb_index, :] = tmp_pose

                print("result of ", uwb_index, self.FusingResult[uwb_index, :])
                print("time used :", time.time() - time_last)
                time_last = time.time()

                '''
                Resample
                '''

                tmp_list = copy.deepcopy(self.ekf_list)
                tmp_weight = self.weight.copy()

                self.ekf_list.clear()

                for i in range(len(tmp_list)):
                    if np.isnan(np.sum(tmp_weight)):
                        tmp_weight = np.ones_like(tmp_weight)
                        tmp_weight /= np.sum(tmp_weight)
                    tmp_rnd = np.random.uniform(0.0, np.sum(tmp_weight))

                    i_index = -1
                    while tmp_rnd > 0.0:
                        i_index += 1
                        tmp_rnd -= tmp_weight[i_index]

                    self.ekf_list.append(copy.deepcopy(tmp_list[i_index]))
                    self.weight[i] = tmp_weight[i_index]

                uwb_index += 1
            else:
                '''
                Sample
                '''
                for i in range(len(self.ekf_list)):
                    self.ekf_list[i].GetPosition(
                        self.ImuData[imu_index, 1:7] +
                        np.random.normal(0.0, self.Sigma, [6]),
                        self.Zupt[imu_index]
                    )
                imu_index += 1


        # import time

        fig = plt.figure(1)
        plt.title("FUSING RESULT(red - fusing,blue - uwb")
        plt.plot(self.FusingResult[:, 0], self.FusingResult[:, 1], 'r-+')
        plt.plot(self.UwbResult[:, 1], self.UwbResult[:, 2], 'b-+')

        plt.grid(True)
        fig.savefig("{0}-{1}-{2}-{3}.png".format(particle_number,
                                                 self.Sigma,
                                                 self.EvaluateSigma,
                                                 time.time()))



if __name__ == '__main__':
    dir_name = './tmp_file_dir/'

    for file_name in os.listdir('./tmp_file_dir/'):
        if 'beaconset' in file_name and (not ('.csv' in file_name)):
            beaconset = np.loadtxt(dir_name + file_name)
            np.savetxt(dir_name + file_name + '.csv', np.loadtxt(dir_name + file_name), delimiter=',')
        elif 'UwbData' in file_name and (not ('.csv' in file_name)):
            UwbData = np.loadtxt(dir_name + file_name)
            np.savetxt(dir_name + file_name + '.csv', np.loadtxt(dir_name + file_name), delimiter=',')

        elif 'UwbResult' in file_name and (not ('.csv' in file_name)):
            UwbResult = np.loadtxt(dir_name + file_name)
            np.savetxt(dir_name + file_name + '.csv', np.loadtxt(dir_name + file_name), delimiter=',')

        elif 'ImuData' in file_name and (not ('.csv' in file_name)):
            ImuData = np.loadtxt(dir_name + file_name)
            np.savetxt(dir_name + file_name + '.csv', np.loadtxt(dir_name + file_name), delimiter=',')

        elif 'Zupt' in file_name and (not ('.csv' in file_name)):
            Zupt = np.loadtxt(dir_name + file_name)
            np.savetxt(dir_name + file_name + '.csv', np.loadtxt(dir_name + file_name), delimiter=',')

        elif 'ImuResultData' in file_name and (not ('.csv' in file_name)):
            ImuResult = np.loadtxt(dir_name + file_name)
            np.savetxt(dir_name + file_name + '.csv', np.loadtxt(dir_name + file_name), delimiter=',')

    # from ViewerModel import PF_FRAME
    #
    # pf_test = PF_FRAME.PF_Frame([11111,1111],[12,12],1.2,1000)
    #
    # pf_test.SetBeaconSet(beaconset[:,0:2])
    #

    '''
    Save file as csv
    '''



    test_ekfpf = EkfPf(beaconset, UwbData, UwbResult,
                       ImuData, Zupt,
                       1000)

    plt.show()
