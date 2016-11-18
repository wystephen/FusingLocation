# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 15　19:43


import numpy as np

import scipy as sp
from scipy.optimize import minimize

import matplotlib.pyplot as plt


class ReferenceTransform:
    def __init__(self):
        self.theta = 0.0
        self.offset = np.asarray([0.0, 0.0])

    def SetOffset(self, offset):
        self.offset = offset

    def SetTheta(self, theta):
        self.theta = theta

    def EstimateTheta(self, pointlist, reference_pointlist):
        '''

        :param pointlist:
        :param referent_theta:
        :return:
        '''
        # print(pointlist.shape)
        # print(referent_vec.shape)
        # print("---------------")

        # ````
        self.imu_path = pointlist[:, 0:2]
        self.uwb_path = reference_pointlist[:, -2:]
        self.imu_path += self.offset

        # print(self.imu_path.shape)
        # print(self.uwb_path.shape)

        #
        plt.figure(1123)
        plt.title("imu before and after transform")
        plt.plot(self.imu_path[:, 0], self.imu_path[:, 1], 'r-+')
        plt.plot(self.uwb_path[:, 0], self.uwb_path[:, 1], 'b-+')
        plt.grid(True)

        init_theta_pose = [0.0,
                           0.0, 0.0]  # 90.0 * np.pi / 180.0

        # init_theta_pose = [180 * np.pi / 180.0 + 135.0 * np.pi / 180.0, 0.0, 0.0]  # 90.0 * np.pi / 180.0
        res = minimize(self.theta_costfunc,
                       init_theta_pose,
                       method='L-BFGS-B',
                       bounds=((-np.pi, np.pi),
                               (-30, 30),
                               (-30, 30)),
                       jac=False)
        # print(res.x)
        self.theta = res.x[0]
        tmp_imu_path = self.Transform(self.imu_path) + res.x[1:3]
        plt.plot(tmp_imu_path[:, 0], tmp_imu_path[:, 1], 'g-+')
        for i in range(0, tmp_imu_path.shape[0], 15):
            plt.plot([tmp_imu_path[i, 0], self.uwb_path[i, 0]],
                     [tmp_imu_path[i, 1], self.uwb_path[i, 1]],
                     'y-')
        self.tmp_imu_path = tmp_imu_path

    def theta_costfunc(self, thetapose):
        theta = thetapose[0]
        pose = thetapose[1:3]
        val = 0.0
        '''
        Check the value range of theta.
        '''
        if not (-np.pi < self.theta < np.pi):
            print("self.theta is out of range:", self.theta)

        '''
        Compute tMatrix
        '''
        tMatrix = np.asarray([
            np.cos(theta), np.sin(theta),
            -np.sin(theta), np.cos(theta)
        ], dtype=float)

        tMatrix = tMatrix.reshape([2, 2])

        tmp_imu = tMatrix.dot(self.imu_path.transpose()).transpose()
        tmp_imu += pose

        val = np.sum((tmp_imu[:tmp_imu.shape[0], :] -
                      self.uwb_path[:tmp_imu.shape[0], :]) ** 2.0)
        # print("val:", val, "thetapose:", thetapose)
        return val

    def ComputeRefOdo(self, odovec, uwb_list, imu_list):
        '''

        :param odovec:
        :param uwb_list:
        :param imu_list:
        :return:
        '''
        self.imu_path_odo = imu_list
        self.uwb_path_odo = uwb_list

        theta_list = list()

        for t_i in range(5):
            ini_thetapose = np.random.normal(0.0, 0.4, size=[3])

            res = minimize(self.odo_costfunc,
                           ini_thetapose,
                           method='L-BFGS-B',
                           bounds=((-np.pi, np.pi),
                                   (-30, 30),
                                   (-30, 30)),
                           jac=False)
            theta_list.append(res.x[0])

        self.theta = np.mean(np.asarray(theta_list))
        # print("theta list:", theta_list)
        print("src and res norm:", np.linalg.norm(odovec),
              np.linalg.norm(self.Transform(odovec)))
        return self.Transform(odovec)

    def odo_costfunc(self, thetapose):
        theta = thetapose[0]
        pose = thetapose[1:3]
        val = 0.0
        '''
        Check the value range of theta.
        '''
        if not (-np.pi < self.theta < np.pi):
            print("self.theta is out of range:", self.theta)

        '''
        Compute tMatrix
        '''
        tMatrix = np.asarray([
            np.cos(theta), np.sin(theta),
            -np.sin(theta), np.cos(theta)
        ], dtype=float)

        tMatrix = tMatrix.reshape([2, 2])

        tmp_imu = tMatrix.dot(self.imu_path_odo.transpose()).transpose()
        tmp_imu += pose

        val = np.sum((tmp_imu -
                      self.uwb_path_odo) ** 2.0)
        # print("val:", val, "thetapose:", thetapose)
        return val

    def Transform(self, pointlist):

        tMatrix = np.zeros([2, 2])

        try:
            '''
            Check the value range of theta.
            '''
            if not (-np.pi < self.theta < np.pi):
                print("self.theta is out of range:", self.theta)

            '''
            Compute tMatrix
            '''
            tMatrix = np.asarray([
                [np.cos(self.theta), np.sin(self.theta)],
                [-np.sin(self.theta), np.cos(self.theta)]
            ], dtype=float)
            tMatrix = tMatrix.reshape([2, 2])

            '''
            Check Order and transform
            '''
            pointlist = tMatrix.dot(pointlist.transpose()).transpose()
            # pointlist += self.offset

        finally:
            # print(tMatrix)
            aaaaaaa = 1111111


        return pointlist
