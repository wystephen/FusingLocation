# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 15　19:43


import numpy as np

import scipy as sp


class reftransform:
    def __init__(self):
        self.theta = 0.0
        self.offset = np.asarray([0.0, 0.0])

    def SetOffset(self, offset):
        self.offset = offset

    def SetTheta(self, theta):
        self.theta = theta

    def EstimateTheta(self,pointlist,referent_vec):
        '''

        :param pointlist:
        :param referent_theta:
        :return:
        '''
        print(pointlist.shape)
        print(referent_vec.shape)
        print("---------------")




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
            pointlist += self.offset



        finally:
            print(tMatrix)
