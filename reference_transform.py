# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 15　19:43


import numpy as np

import scipy as sp
from scipy.optimize import minimize

import matplotlib.pyplot as plt
class reftransform:
    def __init__(self):
        self.theta = 0.0
        self.offset = np.asarray([0.0, 0.0])

    def SetOffset(self, offset):
        self.offset = offset

    def SetTheta(self, theta):
        self.theta = theta

    def EstimateTheta(self,pointlist,reference_pointlist):
        '''

        :param pointlist:
        :param referent_theta:
        :return:
        '''
        # print(pointlist.shape)
        # print(referent_vec.shape)
        # print("---------------")

        #````
        self.imu_path = pointlist[:, 0:2]
        self.uwb_path = reference_pointlist[:,-2:]
        self.imu_path += self.offset

        # print(self.imu_path.shape)
        # print(self.uwb_path.shape)


        plt.figure(1123)
        plt.plot(self.imu_path[:,0],self.imu_path[:,1],'r-+')
        plt.plot(self.uwb_path[:,0],self.uwb_path[:,1],'b-+')
        plt.grid(True)

        init_theta = 0.0
        res = minimize(self.theta_costfunc,
                       init_theta,
                       method='L-BFGS-B',
                       # bounds=((-np.pi,np.pi)),
                       jac=False)
        print(res.x)
        self.theta = res.x
        tmp_imu_path = self.Transform(self.imu_path)
        plt.plot(tmp_imu_path[:, 0], tmp_imu_path[:, 1], 'g-+')

    def theta_costfunc(self, theta):
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
            [np.cos(self.theta), np.sin(self.theta)],
            [-np.sin(self.theta), np.cos(self.theta)]
        ], dtype=float)

        tmp_imu = tMatrix.dot(self.imu_path.transpose()).transpose()

        val = np.sum(np.abs(tmp_imu[0:20, :] - self.uwb_path[0:20, :]))

        return val

    def Transform(self, pointlist):

        tMatrix = np.zeros([2, 2])

        try:
            '''
            Check the value range of theta.
            '''
            if not(-np.pi < self.theta < np.pi):
                print("self.theta is out of range:",self.theta)

            '''
            Compute tMatrix
            '''
            tMatrix = np.asarray([
                [np.cos(self.theta), np.sin(self.theta)],
                [-np.sin(self.theta), np.cos(self.theta)]
            ], dtype=float)

            '''
            Check Order and transform
            '''
            pointlist = tMatrix.dot(pointlist.transpose())
            # pointlist += self.offset



        finally:
            print(tMatrix)

        return pointlist
