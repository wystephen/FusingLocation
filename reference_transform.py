# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 15　19:43


import numpy as np

import scipy as sp


class reftransform:
    def __init__(self):
        self.theta = 0.0
        self.offset = np.asarray([0.0,0.0])

    def SetOffset(self,offset):
        self.offset = offset

    def SetTheta(self,theta):
        self.theta = theta

    def Transform(self,pointlist):

        tMatrix = np.zeros([2,2])

        try:
            tMatrix = np.asarray([
                [np.cos(self.theta),np.sin(self.theta)],
                [-np.sin(self.theta),np.cos(self.theta)]
            ],dtype=float)

        finally:
            print(tMatrix)
